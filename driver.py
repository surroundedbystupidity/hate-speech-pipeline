from builder_v2 import (
    load_and_prepare_data,
    build_node_mappings,
    build_temporal_graph_local_diffusion,
)
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

from temporal_models import BasicRecurrentGCN

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
WINDOW_SIZE_HOURS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def get_graph(author2idx, subreddit2idx, num_subreddits, df) -> DynamicGraphTemporalSignal:
    train_edge_indices, train_edge_weights, train_features, train_labels, train_time_vs_count, train_masks = build_temporal_graph_local_diffusion(
        df=df,
        author2idx=author2idx,
        subreddit2idx=subreddit2idx,
        num_subreddits=num_subreddits,
        comment_feature_names=COMMENT_FEATURE_NAMES,
        user_feature_names=USER_FEATURE_NAMES,
        tox_thresh=0.5
    )
    print("Built training graph with %d snapshots." % len(train_edge_indices))
    return DynamicGraphTemporalSignal(
        edge_indices = train_edge_indices,
        edge_weights = train_edge_weights,
        features = train_features,
        targets = train_labels,
        masks = train_masks
    )

# TODO: Bring plot_time_vs_count here.

def train_model(train_dataset: DynamicGraphTemporalSignal, epochs=10):
    model = BasicRecurrentGCN(
        node_features=train_dataset.num_features,
        hidden_dim=128,
        dropout=0.1,
        num_heads=8
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for train_snapshot in train_dataset:
            model.train()
            optimizer.zero_grad()
            output = model(train_snapshot.x.to(DEVICE), train_snapshot.edge_index.to(DEVICE))
            loss = criterion(output.view(-1), train_snapshot.y.to(DEVICE).view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch}, Average Epoch Loss: {epoch_loss / train_dataset.snapshot_count}")
    return model, criterion

def evaluate_model(model, test_dataset: DynamicGraphTemporalSignal, criterion):
    model.eval()
    # TODO: Factor in masks.
    with torch.no_grad():
        for snapshot in tqdm(test_dataset):
            output = model(snapshot)
            loss = criterion(output, snapshot.targets)
            print(f"Test Loss: {loss.item()}")

def run():
    train_file_path = "val_dataset_with_emb.csv"
    test_file_path = "test_dataset_with_emb_sm.csv"
    print("Preparing windows for %s hours.", WINDOW_SIZE_HOURS)
    df_train = load_and_prepare_data(train_file_path, WINDOW_SIZE_HOURS)
    df_test = load_and_prepare_data(test_file_path, WINDOW_SIZE_HOURS)

    # Build node mappings from combined data to ensure consistency
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    author2idx, subreddit2idx, num_subreddits = build_node_mappings(df_combined)

    train_dataset = get_graph(author2idx, subreddit2idx, num_subreddits, df_train)
    test_dataset = get_graph(author2idx, subreddit2idx, num_subreddits, df_test)

    model = train_model(train_dataset)
    evaluate_model(model, test_dataset)

run()