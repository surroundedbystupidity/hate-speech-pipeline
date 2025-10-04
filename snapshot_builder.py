import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.signal import (
    DynamicGraphTemporalSignal,
    temporal_signal_split,
)
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
GENERATE_EMBEDDINGS = False
logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(module)s(%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim=32):
        super().__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        out = self.fc(h)
        return out


def load_and_prepare_data(csv_path, window_size_hours=0.5):
    df = pd.read_csv(csv_path).dropna().sort_values(by="created_utc").tail(5000)
    df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s")
    df = df.sort_values("timestamp")
    df["time_bin"] = df["timestamp"].dt.floor(f"{window_size_hours}h")
    if GENERATE_EMBEDDINGS:
        df["body_emb"] = list(generate_comment_embeddings(df, "body"))
    else:
        # Convert list of embeddings as a string into a float vector.
        df["body_emb"] = (
            df["body_emb"]
            .str.translate(str.maketrans({"[": "", "]": "", "\n": " "}))
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        # Then vectorized parse
        df["body_emb"] = df["body_emb"].apply(
            lambda s: np.fromstring(s, sep=" ").tolist()
        )
    return df


def build_node_mappings(df):
    # authors = df["author"].unique()
    comments = df["id"].unique()
    subreddits = df["subreddit"].unique()
    node2idx = {}
    idx = 0
    # for author in authors:
    #     node2idx[f"author_{author}"] = idx
    #     idx += 1
    for comment in comments:
        node2idx[f"comment_{comment}"] = idx
        idx += 1
    subreddit2idx = {sub: idx for idx, sub in enumerate(subreddits)}
    return node2idx, subreddit2idx, len(subreddits)


def build_temporal_graph(
    df,
    node2idx,
    subreddit2idx,
    num_subreddits,
    comment_feature_names,
    user_feature_names,
):
    classes2idx = {"non-toxic": 0, "toxic": 1}
    time_groups = df.groupby("time_bin")
    edge_index_list = []
    edge_weight_list = []
    features_list = []
    labels_list = []
    time_vs_count = {}

    for time_bin, group in time_groups:
        num_nodes = group.shape[0]  # now should only contain comment nodes

        logger.debug("Processing time window: %s (%d comments)", time_bin, num_nodes)
        time_vs_count[str(time_bin)] = len(group)
        edges = []
        edge_weights = []

        feature_dim = (
            4
            + num_subreddits
            + len(comment_feature_names)
            + len(user_feature_names)
            + 1  # For author_id
            + 384  # Body text embeddings
        )
        x = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        y = np.zeros((num_nodes, 1), dtype=np.float32)

        for row in group.itertuples(index=False):
            comment_idx = node2idx[f"comment_{row.id}"] % num_nodes
            class_idx = classes2idx[str(row.class_self)]
            subreddit_feature_idx = subreddit2idx[row.subreddit]

            # connect comment → parent comment if available
            if pd.notna(row.parent_id) and f"comment_{row.parent_id}" in node2idx:
                parent_idx = node2idx[f"comment_{row.parent_id}"]
                edges.append([comment_idx, parent_idx])
                edge_weights.append(1.0)

            # ----------------------------
            # Build comment node features
            # ----------------------------
            feat_offset = 0

            # Encode author_id (as numeric feature, could later one-hot/embed)
            x[comment_idx, feat_offset] = (
                float(hash(row.author) % 1e6) / 1e6
            )  # scaled ID
            feat_offset += 1

            # Author/user-level features
            for i, user_feat_name in enumerate(user_feature_names):
                x[comment_idx, feat_offset + i] = getattr(row, user_feat_name)
            feat_offset += len(user_feature_names)

            # Score
            x[comment_idx, feat_offset] = float(row.score_f)
            feat_offset += 1

            # Body length
            x[comment_idx, feat_offset] = len(str(row.body))
            feat_offset += 1

            # Class label (toxic / non-toxic flag)
            x[comment_idx, feat_offset] = float(class_idx)
            feat_offset += 1

            # Extra comment features
            for i, comment_feat_name in enumerate(comment_feature_names):
                x[comment_idx, feat_offset + i] = getattr(row, comment_feat_name)
            feat_offset += len(comment_feature_names)

            # Body embedding (384-dim)
            x[comment_idx, feat_offset : feat_offset + 384] = np.array(
                row.body_emb, dtype=np.float32
            )
            feat_offset += 384

            # Subreddit one-hot
            x[comment_idx, feat_offset + subreddit_feature_idx] = 1.0
            feat_offset += num_subreddits

            # Label (toxicity probability)
            y[comment_idx, 0] = row.toxicity_probability_self

        if edges:
            edge_index = np.array(edges, dtype=np.int64).T.copy()
            edge_weight = np.array(edge_weights, dtype=np.float32).copy()
            edge_index_list.append(edge_index)
            edge_weight_list.append(edge_weight)
            features_list.append(x)
            labels_list.append(y)

    return edge_index_list, edge_weight_list, features_list, labels_list, time_vs_count


def plot_time_vs_count(time_vs_count, step=24):
    dates = list(time_vs_count.keys())
    counts = list(time_vs_count.values())
    plt.plot(dates, counts, marker="o")
    xtick_positions = list(range(0, len(dates), step))
    xtick_labels = [dates[i] for i in xtick_positions]
    plt.xticks(xtick_positions, xtick_labels, rotation=45, ha="right")
    plt.xlabel("Time Window")
    plt.ylabel("Comment Count")
    plt.title("Comment Counts per Time Window")
    plt.tight_layout()
    plt.show()


def generate_comment_embeddings(df, col_name, batch_size=96):
    comments = df[col_name].to_list()
    loader = DataLoader(comments, batch_size=batch_size)

    all_embeddings = []
    for batch in tqdm(loader, desc="Generating embeddings"):
        inputs = TOKENIZER(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            outputs = MODEL(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings).numpy()


def train_and_evaluate(dataset, node_features=1, epochs=50, train_ratio=0.25):
    train_dataset, test_dataset = temporal_signal_split(
        dataset, train_ratio=train_ratio
    )

    train_cmt_count = 0
    test_cmt_count = 0
    for i in range(train_dataset.snapshot_count):
        train_cmt_count += train_dataset[i].x.shape[0]

    for i in range(test_dataset.snapshot_count):
        test_cmt_count += test_dataset[i].x.shape[0]

    logger.info(
        "Train - snapshots = %s, comments = %s",
        train_dataset.snapshot_count,
        train_cmt_count,
    )
    logger.info(
        "Test - snapshots = %s, comments = %s",
        test_dataset.snapshot_count,
        test_cmt_count,
    )

    model = RecurrentGCN(node_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    model.train()
    min_loss = 1
    best_epoch = 0
    for epoch in tqdm(range(epochs), "Epochs"):
        epoch_loss = 0
        for train_snapshot in train_dataset:
            train_snapshot = train_snapshot.to(DEVICE)
            optimizer.zero_grad()
            y_hat = model(
                train_snapshot.x,
                train_snapshot.edge_index,
            )
            loss = F.mse_loss(y_hat, train_snapshot.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if min_loss > epoch_loss:
            min_loss = epoch_loss
            best_epoch = epoch + 1
    logger.info(
        "Best Epoch %d, Train Loss: %.4f",
        best_epoch,
        min_loss / train_dataset.snapshot_count,
    )

    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for snapshot in test_dataset:
            snapshot = snapshot.to(DEVICE)
            y_hat = model(snapshot.x, snapshot.edge_index)
            snapshot_loss = F.mse_loss(y_hat, snapshot.y).item()
            test_loss += snapshot_loss
            logger.debug("Snapshot Loss = %s", snapshot_loss)
    test_loss /= test_dataset.snapshot_count
    save_predictions_to_csv(test_dataset, model, DEVICE)
    logger.info("Test Loss: %.4f", test_loss)


def save_predictions_to_csv(
    test_dataset, model, device, output_path="test_predictions.csv"
):
    all_preds = []
    all_truths = []
    all_indices = []
    for snapshot in test_dataset:
        y_hat = model(snapshot.x.to(device), snapshot.edge_index.to(device))
        preds = y_hat.detach().cpu().numpy().flatten()
        truths = snapshot.y.detach().cpu().numpy().flatten()
        indices = np.arange(len(preds))
        all_preds.extend(preds)
        all_truths.extend(truths)
        all_indices.extend(indices)
    df_out = pd.DataFrame(
        {
            "node_index": all_indices,
            "prediction": all_preds,
            "ground_truth": all_truths,
        }
    )

    df_out["residual"] = df_out["ground_truth"] - df_out["prediction"]
    df_out["abs_residual"] = df_out["residual"].abs()
    mse = mean_squared_error(df_out["ground_truth"], df_out["prediction"])
    logger.info("MSE = %.4f", mse)
    df_out.to_csv(output_path, index=False)
    logger.info("Saved test predictions to %s", output_path)


def main():

    csv_path = "val_with_embeddings.csv"
    window_size_hours = 1
    plot_time_graph = False
    comment_feature_names = [
        "toxicity_probability_self",
        "toxicity_probability_parent",
        "thread_depth",
        "score_f",
        "score_z",
        "scorebin_0",
        "scorebin_2",
        "scorebin_3",
        "scorebin_4",
        "score_bin5",
        "response_time",
        "score_parent",
        "hate_score_self",
        "hate_score_ctx",
    ]
    user_feature_names = [
        "user_unique_subreddits",
        "user_total_comments",
        "user_hate_comments",
        "user_hate_ratio",
        "user_avg_posting_intervall",
        "user_avg_comment_time_of_day",
        "user_hate_comments_ord",
        "user_hate_ratio_ord",
    ]
    df = load_and_prepare_data(csv_path, window_size_hours)
    node2idx, subreddit2idx, num_subreddits = build_node_mappings(df)
    edge_index_list, edge_weight_list, features_list, labels_list, time_vs_count = (
        build_temporal_graph(
            df,
            node2idx,
            subreddit2idx,
            num_subreddits,
            comment_feature_names,
            user_feature_names,
        )
    )
    dataset = DynamicGraphTemporalSignal(
        edge_indices=edge_index_list,
        edge_weights=edge_weight_list,
        features=features_list,
        targets=labels_list,
    )
    logger.info("Dataset created with %d snapshots", dataset.snapshot_count)
    logger.info("Number of nodes: %d", dataset.features[0].shape[0])
    first_snapshot = dataset[0]
    logger.info(
        "First snapshot - Edges: %d, Features: %s, Labels: %s",
        first_snapshot.edge_index.shape[1],
        first_snapshot.x.shape,
        first_snapshot.y.shape,
    )
    if plot_time_graph:
        plot_time_vs_count(time_vs_count, step=24)
    train_and_evaluate(
        dataset,
        node_features=dataset.features[0].shape[1],
        epochs=10,
    )


if __name__ == "__main__":
    main()
