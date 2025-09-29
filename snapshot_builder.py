import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

logger = logging.getLogger(__name__)
logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(module)s(%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
df = pd.read_csv(
    "/Users/sujay/Library/CloudStorage/OneDrive-TheUniversityofAuckland/Course Documents/COMPSCI 760/Project/COMPSCI 760 - Group Project/Sample Data/addNew/retrain_validation10.csv"
).dropna()

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
logger.info("Total nodes: %d (authors + comments)", num_nodes)
logger.info("Total subreddits as features: %d", num_subreddits)


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
    "user_hate_comments_ord",
    "user_hate_ratio_ord",
]

time_vs_count = dict()

for time_bin, group in time_groups:
    logger.info("Processing time window: %s (%d comments)", time_bin, len(group))
    time_vs_count[str(time_bin)] = len(group)

    # Collect all edges for this time window
    edges = []
    edge_weights = []

    # Initialize features and labels for this snapshot
    # Features: [node_type_author, node_type_comment, score, text_length, subreddit_onehot...]
    feature_dim = (
        4 + num_subreddits + len(comment_feature_names) + len(user_feature_names)
    )  # Base features + one-hot subreddit encoding
    x = np.zeros((num_nodes, feature_dim), dtype=np.float32)
    y = np.zeros((num_nodes, 1), dtype=np.float32)

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

        # Add user features to author node
        for i, user_feat_name in enumerate(user_feature_names):
            logger.debug(
                "%s feature name = %s at index (%s, %s)",
                "author",
                user_feat_name,
                author_idx,
                i + 1,
            )
            x[author_idx, i + 1] = row[user_feat_name]

        # Comment indicator on node
        x[comment_idx, len(user_feature_names) + 1] = 1.0
        # Comment score
        x[comment_idx, len(user_feature_names) + 2] = float(row.score_f)
        # Text length
        x[comment_idx, len(user_feature_names) + 3] = (
            len(str(row.body)) if pd.notna(row.body) else 0
        )
        # Toxic/non-toxic
        x[comment_idx, len(user_feature_names) + 4] = float(class_idx)

        # Set all comment features.
        for i, comment_feat_name in enumerate(comment_feature_names):
            logger.debug(
                "%s feature name = %s at index (%s, %s)",
                "comment",
                comment_feat_name,
                comment_idx,
                len(user_feature_names) + 4 + i + 1,
            )
            x[comment_idx, len(user_feature_names) + 4 + i + 1] = row[comment_feat_name]

        # Subreddit as one-hot feature for comment (starting at index 4)
        x[
            comment_idx,
            len(user_feature_names)
            + len(comment_feature_names)
            + subreddit_feature_idx,
        ] = 1.0

        # Set labels (hate detection target)
        y[comment_idx, 0] = float(row.toxicity_probability_self)

        if edges:
            edge_index = np.array(edges, dtype=np.int64).T.copy()
            edge_weight = np.array(edge_weights, dtype=np.float32).copy()
            edge_index_list.append(edge_index)
            edge_weight_list.append(edge_weight)
            features_list.append(x)
            labels_list.append(y)

logger.info("Created %d temporal snapshots", len(edge_index_list))

dataset = DynamicGraphTemporalSignal(
    edge_indices=edge_index_list,
    edge_weights=edge_weight_list,
    features=features_list,
    targets=labels_list,
)

# Verify the dataset
logger.info("Dataset created with %d snapshots", dataset.snapshot_count)
logger.info("Node feature dimension: %s", dataset.features[0].shape)
logger.info("Number of nodes: %d", dataset.features[0].shape[0])

# Example: Print first snapshot info
first_snapshot = dataset[0]

logger.info(
    "First snapshot - Edges: %d, Features: %s, Labels: %s",
    first_snapshot.edge_index.shape[1],
    first_snapshot.x.shape,
    first_snapshot.y.shape,
)

dates = list(time_vs_count.keys())
counts = list(time_vs_count.values())

plt.plot(dates, counts, marker="o")

# Show only every 12th label
step = 24
xtick_positions = list(range(0, len(dates), step))
xtick_labels = [dates[i] for i in xtick_positions]
plt.xticks(xtick_positions, xtick_labels, rotation=45, ha="right")

plt.xlabel("Time Window")
plt.ylabel("Comment Count")
plt.title("Comment Counts per Time Window")
plt.tight_layout()
plt.show()
