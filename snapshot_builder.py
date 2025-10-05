import glob
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from torch_geometric_temporal.nn import DCRNN
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from loss_functions import FocalLoss, HybridLoss, WeightedBCELoss

DEVICE = "mps"  # torch.device("mps" if torch.mps.is_available() else "cpu")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATE_EMBEDDINGS = False
PRINT_CORRELATION = False
WINDOW_SIZE_HOURS = 1
PLOT_TIME_GRAPH = False
COMMENT_FEATURE_NAMES: list[str] = [
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
USER_FEATURE_NAMES: list[str] = [
    "user_unique_subreddits",
    "user_total_comments",
    "user_hate_comments",
    "user_hate_ratio",
    "user_avg_posting_intervall",
    "user_avg_comment_time_of_day",
    "user_hate_comments_ord",
    "user_hate_ratio_ord",
]

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(module)s(%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """Self-attention layer for graph nodes"""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [num_nodes, hidden_dim]
        x_unsqueezed = x.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        attended, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        attended = attended.squeeze(0)  # [num_nodes, hidden_dim]
        return self.norm(x + self.dropout(attended))


class RecurrentGCN1(nn.Module):
    def __init__(self, node_features, hidden_dim=128, dropout=0.1, num_heads=8):
        super().__init__()
        self.recurrent = DCRNN(node_features, hidden_dim, K=1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        attn_out, _ = self.attention(h, h, h)
        h = h + attn_out
        out = self.fc(h)
        return out


class RecurrentGCN(nn.Module):
    """Enhanced model with attention and deeper architecture"""

    def __init__(self, node_features, hidden_dim=256, dropout=0.15, num_gnn_layers=2):
        super().__init__()

        # Input transformation
        self.input_transform = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Multiple GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            input_dim = hidden_dim if i > 0 else hidden_dim
            self.gnn_layers.append(DCRNN(input_dim, hidden_dim, K=2))

        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim, num_heads=4, dropout=dropout)

        # Skip connection
        self.skip_transform = nn.Linear(node_features, hidden_dim)

        # Output layers with more capacity
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Layer normalization for stability
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)]
        )

    def forward(self, x, edge_index):
        # Transform input
        h = self.input_transform(x)
        skip = self.skip_transform(x)

        # Process through GNN layers with residual connections
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_new = gnn_layer(h, edge_index)
            h_new = nn.functional.relu(h_new)
            h = norm(h + h_new * 0.5)  # Residual with scaling

        # Apply attention
        h = self.attention(h)

        # Combine with skip connection
        h_combined = torch.cat([h, skip], dim=-1)

        # Generate output
        out = self.output_layers(h_combined)
        return out


def load_and_prepare_data(csv_path, window_size_hours):
    df = pd.read_csv(csv_path).head(5000)

    # Drop NAs and filter subreddit without regex overhead (regex is slower)
    mask_valid = df["subreddit"].notna() & ~df["subreddit"].str.contains(" ")
    df = df[mask_valid & df["created_utc"].notna()]

    # Vectorized datetime conversion
    df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df[df["timestamp"].dt.year >= 2015]

    # Precompute time bins once
    df["time_bin"] = df["timestamp"].dt.floor(f"{window_size_hours}h")

    # Handle embeddings
    if GENERATE_EMBEDDINGS:
        df["body_emb"] = generate_comment_embeddings(df, "body")
        df.to_csv("val_dataset_with_emb.csv", index=False)
    else:
        # String ops
        trans = str.maketrans({"[": "", "]": "", "\n": " "})
        cleaned = (
            df["body_emb"]
            .astype(str)
            .str.translate(trans)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # Convert embeddings string to vector.
        empty = np.array([], dtype=np.float32)
        df["body_emb"] = cleaned.map(
            lambda s: np.fromstring(s, dtype=np.float32, sep=" ") if s else empty
        )

    return df


def build_node_mappings(df):
    authors = df["author"].unique()
    subreddits = df["subreddit"].unique()
    author2idx = {}
    idx = 0
    for author in authors:
        author2idx[f"author_{author}"] = idx
        idx += 1
    subreddit2idx = {sub: idx for idx, sub in enumerate(subreddits)}
    return author2idx, subreddit2idx, len(subreddits)


def build_temporal_graph(
    df,
    author2idx,
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
    masks_list = []
    time_vs_count = {}

    for time_bin, group in time_groups:
        num_nodes = group.shape[0]

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

        local_node2idx = {}
        comment_ids = group["id"].to_list()
        parent_ids = []  # group["parent_id"].unique().tolist()
        node_list = np.unique(comment_ids + parent_ids).tolist()

        for i, c_id in enumerate(node_list):
            local_node2idx[f"comment_{c_id}"] = i

        logger.debug(
            "Comments (%s) + parent nodes (%s) = %s",
            num_nodes,
            len(parent_ids),
            num_nodes + len(parent_ids),
        )
        num_nodes += len(parent_ids)

        mask = np.zeros(num_nodes, dtype=bool)
        for i, c_id in enumerate(node_list):
            if c_id in comment_ids:
                mask[i] = True

        # Feature and label on both parent and child comments, parent comments may be
        # from before this timestep.
        x = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        y = np.zeros((num_nodes, 1), dtype=np.float32)

        parent_and_cmts = df[df["id"].isin(node_list)]
        # parent_and_cmts.drop("body_emb", axis=1).to_csv(
        #     f"parent_and_cmts{datetime.datetime.now()}.csv"
        # )
        logger.debug("Final group shape = %s", parent_and_cmts.shape)
        for row in parent_and_cmts.itertuples(index=False):
            comment_idx = local_node2idx[f"comment_{row.id}"]
            class_idx = classes2idx[str(row.class_self)]
            subreddit_feature_idx = subreddit2idx[row.subreddit]

            # connect comment → parent comment if available
            if pd.notna(row.parent_id) and f"comment_{row.parent_id}" in local_node2idx:
                parent_idx = local_node2idx[f"comment_{row.parent_id}"]
                edges.append([comment_idx, parent_idx])
                edge_weights.append(1.0)

            feat_offset = 0

            x[comment_idx, feat_offset] = author2idx[f"author_{row.author}"]
            feat_offset += 1

            # User features
            for i, user_feat_name in enumerate(user_feature_names):
                x[comment_idx, feat_offset + i] = getattr(row, user_feat_name)
            feat_offset += len(user_feature_names)

            # Score
            x[comment_idx, feat_offset] = float(row.score_f)
            feat_offset += 1

            # Body length
            x[comment_idx, feat_offset] = len(str(row.body))
            feat_offset += 1

            # Class label
            x[comment_idx, feat_offset] = float(class_idx)
            feat_offset += 1

            # Comment features
            for i, comment_feat_name in enumerate(comment_feature_names):
                x[comment_idx, feat_offset + i] = getattr(row, comment_feat_name)
            feat_offset += len(comment_feature_names)

            # Vector embedding (384)
            x[comment_idx, feat_offset : feat_offset + 384] = np.array(
                row.body_emb, dtype=np.float32
            )
            feat_offset += 384

            # Subreddit
            x[comment_idx, feat_offset + subreddit_feature_idx] = 1.0
            feat_offset += num_subreddits

            y[comment_idx, 0] = row.toxicity_probability_self

        if edges:
            edge_index = np.array(edges, dtype=np.int64).T.copy()
            edge_weight = np.array(edge_weights, dtype=np.float32).copy()
            edge_index_list.append(edge_index)
            edge_weight_list.append(edge_weight)
            features_list.append(x)
            labels_list.append(y)
            masks_list.append(mask)

    return (
        edge_index_list,
        edge_weight_list,
        features_list,
        labels_list,
        time_vs_count,
        masks_list,
    )


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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    comments = df[col_name].to_list()
    loader = DataLoader(comments, batch_size=batch_size)

    all_embeddings = []
    for batch in tqdm(loader, desc="Generating embeddings"):
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings).numpy()


def train_model(
    train_dataset: DynamicGraphTemporalSignal, node_features, epochs=50
) -> RecurrentGCN:
    train_masks = train_dataset.masks

    model = RecurrentGCN(node_features=node_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = FocalLoss()

    # Training
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for idx, train_snapshot in enumerate(train_dataset):
            train_snapshot = train_snapshot.to(DEVICE)
            train_snapshot.x = nn.functional.normalize(train_snapshot.x, dim=-1)
            train_mask = torch.tensor(train_masks[idx], dtype=torch.bool, device=DEVICE)

            optimizer.zero_grad()
            y_hat = model(train_snapshot.x, train_snapshot.edge_index)

            # ensure target is float tensor
            loss = criterion(
                y_hat[train_mask].view(-1),
                train_snapshot.y[train_mask].float().view(-1),
            )

            num_masked = train_mask.sum().item()
            logger.debug(
                "Cutting TRAIN loss by a factor of %s in %s observations.",
                num_masked,
                len(train_snapshot.y),
            )
            normalized_loss = loss  # / num_masked

            normalized_loss.backward()

            # loss = criterion(
            #     y_hat[train_mask].view(-1), train_snapshot.y[train_mask].view(-1)
            # )
            # loss = loss / accumulation_steps  # Scale loss
            # loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += normalized_loss.item()

        logger.info("Epoch %03d | Loss: %.4f", epoch + 1, epoch_loss)

    return model


def evaluate_model(model: RecurrentGCN, test_dataset: DynamicGraphTemporalSignal):
    test_masks = test_dataset.masks
    # Evaluation
    model.eval()
    criterion = FocalLoss()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx, test_snapshot in enumerate(test_dataset):
            test_loss = 0.0
            test_snapshot = test_snapshot.to(DEVICE)
            test_snapshot.x = nn.functional.normalize(test_snapshot.x, dim=-1)
            test_mask = torch.tensor(test_masks[idx], dtype=torch.bool, device=DEVICE)
            y_hat = model(test_snapshot.x, test_snapshot.edge_index)  # raw logits
            # loss = criterion(
            #     y_hat[test_mask].view(-1), test_snapshot.y[test_mask].float().view(-1)
            # )

            loss = criterion(
                y_hat[test_mask].view(-1), test_snapshot.y[test_mask].view(-1)
            )
            # Normalize by number of masked nodes
            num_masked = test_mask.sum().item()
            logger.debug(
                "Cutting TEST loss by a factor of %s in %s observations.",
                num_masked,
                len(test_snapshot.y),
            )
            normalized_loss = loss  # / num_masked

            test_loss += normalized_loss.item()

            # Convert logits to probabilities
            probs = torch.sigmoid(y_hat).view(-1).cpu()
            labels = test_snapshot.y.view(-1).cpu()

            all_preds.append(probs)
            all_labels.append(labels)

            logger.debug(
                "Snapshot Loss = %.4f | Pred Range: [%.3f, %.3f]",
                loss.item(),
                probs.min().item(),
                probs.max().item(),
            )

    # Concatenate predictions for saving
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Example: save predictions
    save_predictions_to_csv(all_preds, all_labels)

    logger.info(
        "Pred Range (All): [%.3f, %.3f], Mean: %.3f",
        all_preds.min().item(),
        all_preds.max().item(),
        all_preds.mean().item(),
    )


def save_predictions_to_csv(all_preds, all_labels, output_path="test_predictions.csv"):
    df_out = pd.DataFrame(
        {
            "node_index": np.arange(len(all_preds)),
            "prediction": all_preds,
            "ground_truth": all_labels,
        }
    )
    df_out["residual"] = df_out["ground_truth"] - df_out["prediction"]
    df_out["abs_residual"] = df_out["residual"].abs()
    mse = mean_squared_error(df_out["ground_truth"], df_out["prediction"])
    logger.info("MSE = %.4f", mse)
    df_out.to_csv(output_path, index=False)
    logger.info("Saved test predictions to %s", output_path)


def get_correlation(df: pd.DataFrame):
    df_analysis = df[COMMENT_FEATURE_NAMES + USER_FEATURE_NAMES].copy()
    correlations = df_analysis.corr()["toxicity_probability_self"].sort_values(
        ascending=False
    )
    logger.info("Feature correlations with target:\n%s", correlations)


def plot_results():
    df_res = pd.read_csv(
        "/Users/sujay/Documents/Workspace/hate-speech-pipeline/test_predictions.csv"
    )

    # Plot histograms for value distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Distribution of Prediction Results", fontsize=16)

    # Prediction histogram
    axes[0, 0].hist(
        df_res["prediction"], bins=50, alpha=0.7, color="blue", edgecolor="black"
    )
    axes[0, 0].set_title("Prediction Distribution")
    axes[0, 0].set_xlabel("Prediction Value")
    axes[0, 0].set_ylabel("Frequency")

    # Ground truth histogram
    axes[0, 1].hist(
        df_res["ground_truth"], bins=50, alpha=0.7, color="green", edgecolor="black"
    )
    axes[0, 1].set_title("Ground Truth Distribution")
    axes[0, 1].set_xlabel("Ground Truth Value")
    axes[0, 1].set_ylabel("Frequency")

    # Residual histogram
    axes[1, 0].hist(
        df_res["residual"], bins=50, alpha=0.7, color="red", edgecolor="black"
    )
    axes[1, 0].set_title("Residual Distribution")
    axes[1, 0].set_xlabel("Residual Value (Ground Truth - Prediction)")
    axes[1, 0].set_ylabel("Frequency")

    # Absolute residual histogram
    axes[1, 1].hist(
        df_res["abs_residual"], bins=50, alpha=0.7, color="orange", edgecolor="black"
    )
    axes[1, 1].set_title("Absolute Residual Distribution")
    axes[1, 1].set_xlabel("Absolute Residual Value")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    logger.info(
        "Summary Statistics: \n %s",
        df_res[["prediction", "ground_truth", "residual", "abs_residual"]].describe(),
    )


def create_dataset(df: pd.DataFrame, author2idx, subreddit2idx, num_subreddits):
    """Create a temporal graph dataset from a dataframe."""
    (
        edge_index_list,
        edge_weight_list,
        features_list,
        labels_list,
        time_vs_count,
        masks_list,
    ) = build_temporal_graph(
        df,
        author2idx,
        subreddit2idx,
        num_subreddits,
        COMMENT_FEATURE_NAMES,
        USER_FEATURE_NAMES,
    )

    dataset = DynamicGraphTemporalSignal(
        edge_indices=edge_index_list,
        edge_weights=edge_weight_list,
        features=features_list,
        targets=labels_list,
        masks=masks_list,
    )
    cmt_count = 0
    logger.info("Dataset created with %d snapshots", dataset.snapshot_count)

    for train_snapshot in dataset:
        cmt_count += train_snapshot.x.shape[0]

    logger.info(
        "Snapshots = %s, comments = %s",
        dataset.snapshot_count,
        cmt_count,
    )
    return dataset, time_vs_count


def run():
    train_csv_path = "val_dataset_with_emb.csv"
    test_csv_path = "test_dataset_with_emb_sm.csv"
    logger.info("Preparing windows for %s hours.", WINDOW_SIZE_HOURS)
    df_train = load_and_prepare_data(train_csv_path, WINDOW_SIZE_HOURS)
    df_test = load_and_prepare_data(test_csv_path, WINDOW_SIZE_HOURS)

    if PRINT_CORRELATION:
        get_correlation(df_train)
        get_correlation(df_test)

    # Build node mappings from combined data to ensure consistency
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    author2idx, subreddit2idx, num_subreddits = build_node_mappings(df_combined)

    # df_train = balance_score_bins(df_train)

    # print(df_train["toxicity_probability_self"].describe())
    # plt.hist(
    #     df_train["toxicity_probability_self"],
    #     bins=50,
    #     alpha=0.7,
    #     color="blue",
    #     edgecolor="black",
    # )
    # plt.tight_layout()
    # plt.show()
    # return

    # Create datasets
    logger.info("** Begin loading: %s", train_csv_path)
    train_dataset, train_time_vs_count = create_dataset(
        df_train, author2idx, subreddit2idx, num_subreddits
    )
    logger.info("** Finished loading: %s", train_csv_path)
    logger.info("** Begin loading: %s", test_csv_path)
    test_dataset, test_time_vs_count = create_dataset(
        df_test, author2idx, subreddit2idx, num_subreddits
    )
    logger.info("** Finished loading: %s", test_csv_path)

    if PLOT_TIME_GRAPH:
        plot_time_vs_count(train_time_vs_count, step=24)
        plot_time_vs_count(test_time_vs_count, step=24)

    # Use train dataset for training (you can modify this to use both datasets as needed)
    trained_model = train_model(
        train_dataset,
        node_features=train_dataset.features[0].shape[1],
        epochs=25,
    )
    evaluate_model(trained_model, test_dataset)


def balance_score_bins(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0.0, 0.20, 0.40, 0.60, 0.80, 1.00]
    labels = ["bin1", "bin2", "bin3", "bin4", "bin5"]

    df["toxicity_bin"] = pd.cut(
        df["toxicity_probability_self"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    df["toxicity_bin"].value_counts().sort_index()
    min_size = df["toxicity_bin"].value_counts().min()
    print(f"Smallest bin size: {min_size}")
    return (
        df.groupby("toxicity_bin", group_keys=False)
        .apply(lambda g: g.sample(n=min_size, random_state=42))
        .reset_index(drop=True)
    )


def read_parent_cmts():
    # Find all CSV files starting with "parent_and_cmts"
    pattern = (
        "/Users/sujay/Documents/Workspace/hate-speech-pipeline/parent_and_cmts*.csv"
    )
    csv_files = glob.glob(pattern)

    logger.info("Found %d parent_and_cmts CSV files", len(csv_files))

    # Read all CSV files and concatenate them
    dataframes = []
    for file_path in csv_files:
        logger.info("Reading file: %s", file_path)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate all dataframes
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.debug("Combined dataframe shape: %s", combined_df.shape)

        # Prediction histogram
        plt.hist(
            combined_df["toxicity_probability_self"],
            bins=50,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        plt.xlabel("toxicity_probability_self")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
        return combined_df
    else:
        logger.warning("No parent_and_cmts CSV files found")


run()
# plot_results()
