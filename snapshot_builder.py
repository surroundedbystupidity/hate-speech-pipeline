import logging

import pandas as pd
import torch
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

logger = logging.getLogger(__name__)
logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(module)s(%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
df = (
    pd.read_csv(
        "/Users/sujay/Library/CloudStorage/OneDrive-TheUniversityofAuckland/Course Documents/COMPSCI 760/Project/COMPSCI 760 - Group Project/Sample Data/addNew/retrain_validation10.csv"
    )
    .dropna()
    .head(100)
)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s")
df = df.sort_values("timestamp")

# ==== Node Encoding ====
# Only authors and comments as nodes, subreddits as features
authors = df["author"].unique()
comments = df["id"].unique()
subreddits = df["subreddit"].unique()
classes2idx = {"non-toxic": 0, "toxic": 1}

# Create subreddit encoding for features
subreddit2idx = {sub: idx for idx, sub in enumerate(subreddits)}
num_subreddits = len(subreddits)

# Create node mappings with type prefixes to avoid conflicts
node2idx = {}
idx = 0

# Add author nodes
for author in authors:
    node2idx[f"author_{author}"] = idx
    idx += 1

# Add comment nodes
for comment in comments:
    node2idx[f"comment_{comment}"] = idx
    idx += 1


num_nodes = len(node2idx)
logger.info(f"Total nodes: {num_nodes} (authors + comments)")
logger.info(f"Total subreddits as features: {num_subreddits}")


# Group comments into hourly windows.
time_windows = []
window_size_hours = 1  # Adjust as needed

df["time_bin"] = df["timestamp"].dt.floor(f"{window_size_hours}h")
time_groups = df.groupby("time_bin")


edge_index_list = []
edge_weight_list = []
features_list = []
labels_list = []
comment_feature_names = [
    "toxicity_probability_self",
    "class_self",
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
    "user_avg_posting_interval",
    "user_avg_comment_time_of_day",
    # "user_hate_comments_ord",
    # "user_hate_ratio_ord",
]
for time_bin, group in time_groups:
    logger.info(f"Processing time window: {time_bin} ({len(group)} comments)")

    # Collect all edges for this time window
    edges = []
    edge_weights = []

    # Initialize features and labels for this snapshot
    # Features: [node_type_author, node_type_comment, score, text_length, subreddit_onehot...]
    feature_dim = (
        4 + num_subreddits + len(comment_feature_names) + len(user_feature_names)
    )  # Base features + one-hot subreddit encoding
    x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
    y = torch.zeros((num_nodes, 1), dtype=torch.float)

    for _, row in group.iterrows():
        author_idx = node2idx[f"author_{row.author}"]
        comment_idx = node2idx[f"comment_{row.id}"]
        class_idx = classes2idx[row.class_self]
        subreddit_feature_idx = subreddit2idx[row.subreddit]

        # 1. Author -> Comment edge
        edges.append([author_idx, comment_idx])
        edge_weights.append(1.0)

        # 2. Comment -> Comment edge (if found)
        if pd.notna(row.parent_id) and f"comment_{row.parent_id}" in node2idx:
            parent_idx = node2idx[f"comment_{row.parent_id}"]
            edges.append([comment_idx, parent_idx])  # Reply relationship
            edge_weights.append(1.0)

        # Author indicator on node
        x[author_idx, 0] = 1.0

        # Comment indicator on node
        x[comment_idx, 1] = 1.0
        # Comment score
        x[comment_idx, 2] = float(row.score_f)
        # Text length
        x[comment_idx, 3] = len(str(row.body)) if pd.notna(row.body) else 0
        # Toxic/non-toxic
        x[comment_idx, 3] = float(class_idx)
        # Subreddit as one-hot feature for comment (starting at index 4)
        x[comment_idx, 4 + subreddit_feature_idx] = 1.0

        # Set labels (hate detection target)
        y[comment_idx, 0] = float(row.toxicity_probability_self)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    edge_index_list.append(edge_index)
    edge_weight_list.append(edge_weight)
    features_list.append(x)
    labels_list.append(y)

logger.info(f"Created {len(edge_index_list)} temporal snapshots")

dataset = DynamicGraphTemporalSignal(
    edge_indices=edge_index_list,
    edge_weights=edge_weight_list,
    features=features_list,
    targets=labels_list,
)

# Verify the dataset
logger.info(f"Dataset created with {dataset.snapshot_count} snapshots")
logger.info(f"Node feature dimension: {dataset.features[0].shape}")
logger.info(f"Number of nodes: {dataset.features[0].shape[0]}")

# Example: Print first snapshot info
first_snapshot = dataset[0]
logger.info(
    f"First snapshot - Edges: {first_snapshot.edge_index.shape[1]}, "
    f"Features: {first_snapshot.x.shape}, "
    f"Labels: {first_snapshot.y.shape}"
)
