import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.signal import (
    DynamicGraphTemporalSignal,
    temporal_signal_split,
)
from tqdm import tqdm

DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h


def load_and_prepare_data(csv_path, window_size_hours=1):
    df = pd.read_csv(csv_path).dropna().head(100)
    df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s")
    df = df.sort_values("timestamp")
    df["time_bin"] = df["timestamp"].dt.floor(f"{window_size_hours}h")
    return df


def build_node_mappings(df):
    authors = df["author"].unique()
    comments = df["id"].unique()
    subreddits = df["subreddit"].unique()
    node2idx = {}
    idx = 0
    for author in authors:
        node2idx[f"author_{author}"] = idx
        idx += 1
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
    time_vs_count = dict()
    num_nodes = len(node2idx)
    for time_bin, group in time_groups:
        logging.debug("Processing time window: %s (%d comments)", time_bin, len(group))
        time_vs_count[str(time_bin)] = len(group)
        edges = []
        edge_weights = []
        feature_dim = (
            4 + num_subreddits + len(comment_feature_names) + len(user_feature_names)
        )
        x = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        y = np.zeros((num_nodes, 1), dtype=np.float32)
        for row in group.itertuples(index=False):
            author_idx = node2idx[f"author_{row.author}"]
            comment_idx = node2idx[f"comment_{row.id}"]
            class_idx = classes2idx[str(row.class_self)]
            subreddit_feature_idx = subreddit2idx[row.subreddit]
            edges.append([author_idx, comment_idx])
            edge_weights.append(1.0)
            if pd.notna(row.parent_id) and f"comment_{row.parent_id}" in node2idx:
                parent_idx = node2idx[f"comment_{row.parent_id}"]
                edges.append([comment_idx, parent_idx])
                edge_weights.append(1.0)
            x[author_idx, 0] = 1.0
            for i, user_feat_name in enumerate(user_feature_names):
                x[author_idx, i + 1] = getattr(row, user_feat_name)
            x[comment_idx, len(user_feature_names) + 1] = 1.0
            x[comment_idx, len(user_feature_names) + 2] = float(row.score_f)
            x[comment_idx, len(user_feature_names) + 3] = (
                len(str(row.body)) if pd.notna(row.body) else 0
            )
            x[comment_idx, len(user_feature_names) + 4] = float(class_idx)
            for i, comment_feat_name in enumerate(comment_feature_names):
                x[comment_idx, len(user_feature_names) + 4 + i + 1] = getattr(
                    row, comment_feat_name
                )
            x[
                comment_idx,
                len(user_feature_names)
                + len(comment_feature_names)
                + subreddit_feature_idx,
            ] = 1.0
            y[comment_idx, 0] = float(row.toxicity_probability_self)
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


def train_and_evaluate(dataset, node_features=1, epochs=50, train_ratio=0.8):
    train_dataset, test_dataset = temporal_signal_split(
        dataset, train_ratio=train_ratio
    )

    model = RecurrentGCN(node_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    model.train()
    min_loss = 1
    best_epoch = 0
    for epoch in tqdm(range(epochs), "Epochs"):
        epoch_loss = 0
        for snapshot in train_dataset:
            optimizer.zero_grad()
            y_hat = model(
                snapshot.x.to(DEVICE),
                snapshot.edge_index.to(DEVICE),
            )
            loss = F.mse_loss(y_hat, snapshot.y.to(DEVICE))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if min_loss > epoch_loss:
            min_loss = epoch_loss
            best_epoch = epoch + 1
    logging.info(
        "Best Epoch %d, Train Loss: %.4f",
        best_epoch,
        min_loss / train_dataset.snapshot_count,
    )

    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for snapshot in tqdm(test_dataset):
            y_hat = model(
                snapshot.x.to(DEVICE),
                snapshot.edge_index.to(DEVICE),
            )
            test_loss += F.mse_loss(y_hat, snapshot.y.to(DEVICE)).item()
    test_loss /= test_dataset.snapshot_count
    logging.info("Test MSE: %.4f", test_loss)


def main():
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(module)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    csv_path = "/Users/sujay/Library/CloudStorage/OneDrive-TheUniversityofAuckland/Course Documents/COMPSCI 760/Project/COMPSCI 760 - Group Project/Sample Data/addNew/AddNew/retrain_validation10.csv"
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
    logging.info("Dataset created with %d snapshots", dataset.snapshot_count)
    logging.info("Node feature dimension: %s", dataset.features[0].shape)
    logging.info("Number of nodes: %d", dataset.features[0].shape[0])
    first_snapshot = dataset[0]
    logging.info(
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
        epochs=50,
        train_ratio=0.2,
    )


if __name__ == "__main__":
    main()
