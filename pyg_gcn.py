import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEVICE = "mps"  # torch.device("mps" if torch.mps.is_available() else "cpu")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATE_EMBEDDINGS = False
PRINT_CORRELATION = False
WINDOW_SIZE_HOURS = 1
PLOT_TIME_GRAPH = False
TRAIN_CSV_PATH = "val_dataset_with_emb_sm.csv"
TEST_CSV_PATH = "test_dataset_with_emb_sm.csv"
COMMENT_FEATURE_NAMES: list[str] = [
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


class GCN(nn.Module):
    def __init__(
        self, num_node_features, hidden_dim=128, output_dimension=1, dropout=0.3
    ):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim // 2, output_dimension)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output with sigmoid for 0-1 range
        x = self.readout(x)
        x = torch.sigmoid(x)
        return x


def generate_comment_embeddings(
    df: pd.DataFrame, col_name: str, batch_size=96
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    emb_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    comments = df[col_name].to_list()
    loader = DataLoader(comments, batch_size=batch_size)  # type: ignore

    all_embeddings = []
    for batch in tqdm(loader, desc="Generating embeddings"):
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            outputs = emb_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings).numpy()


def create_pyg_dataset(
    df: pd.DataFrame,
    feature_columns,
    scaler=None,
    response_column="toxicity_probability_self",
    id_col="id",
    parent_id_col="parent_id",
):
    df_clone = df.copy()

    # Coerce feature columns to numeric
    df_clone[feature_columns] = (
        df_clone[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
    )
    numeric_array = df_clone[feature_columns].values

    # Stack the embedding column into a 2D numpy array
    embeddings_array = np.vstack(df["body_emb"].values.tolist())

    # Concatenate numeric + embedding along feature dimension
    x_array = np.hstack([numeric_array, embeddings_array])

    # Normalize features
    if scaler is None:
        scaler = StandardScaler()
        x_array = scaler.fit_transform(x_array)
    else:
        x_array = scaler.transform(x_array)

    # Convert to tensor
    x = torch.tensor(x_array, dtype=torch.float32)
    y = torch.tensor(df_clone[response_column].values, dtype=torch.float)

    # Map id to node index
    id_to_idx = {id_: idx for idx, id_ in enumerate(df_clone[id_col])}

    # Build edge list: parent -> child
    child_ids = df_clone[id_col].values
    parent_ids = df_clone[parent_id_col].values

    edge_list = [
        (id_to_idx[p], id_to_idx[c])
        for c, p in zip(child_ids, parent_ids)
        if p in id_to_idx
    ]

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, y=y, edge_index=edge_index)
    return data, scaler


def plot_results():
    df_res = pd.read_csv(
        "/Users/sujay/Documents/Workspace/hate-speech-pipeline/gcn_predictions.csv"
    )

    # Plot histograms for value distributions
    fig, axes = plt.subplots(2, 2)
    fig.suptitle("Distribution of Prediction Results")

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

    # Absolute residual histogram
    axes[1, 0].hist(
        df_res["absolute_error"], bins=50, alpha=0.7, color="orange", edgecolor="black"
    )
    axes[1, 0].set_title("Absolute Error Distribution")
    axes[1, 0].set_xlabel("Absolute Error Value")
    axes[1, 0].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    logger.info(
        "Summary Statistics: \n %s",
        df_res[["prediction", "ground_truth", "absolute_error"]].describe(),
    )


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).head(5000)

    # Drop NAs and filter subreddit without regex overhead
    mask_valid = df["subreddit"].notna() & ~df["subreddit"].str.contains(" ")
    df = df[mask_valid & df["created_utc"].notna()]

    # Vectorized datetime conversion
    df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df[df["timestamp"].dt.year >= 2015]

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

        # Convert embeddings string to vector
        empty = np.array([], dtype=np.float32)
        df["body_emb"] = cleaned.map(
            lambda s: np.fromstring(s, dtype=np.float32, sep=" ") if s else empty
        )

    return df


def evaluate_model(model, data, device):
    """Get predictions and calculate metrics"""
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index).cpu().numpy().flatten()
        ground_truth = data.y.cpu().numpy().flatten()

        mse = mean_squared_error(ground_truth, predictions)
        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mse)

        return predictions, ground_truth, mse, mae, rmse


def train():
    if train_data.x is not None:
        gcn_model = GCN(
            num_node_features=train_data.x.shape[1],
            hidden_dim=256,
            output_dimension=1,
            dropout=0.3,
        ).to(DEVICE)

        optimizer = optim.AdamW(gcn_model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Better loss function for bounded regression
        loss_fn = nn.BCELoss()  # Since output is sigmoid-activated and target is [0,1]

        best_loss = float("inf")
        patience_counter = 0
        patience = 15

        gcn_model.train()
        for epoch in range(1, 101):
            optimizer.zero_grad()
            out = gcn_model(train_data.x, train_data.edge_index)
            loss = loss_fn(out.view(-1), train_data.y.view(-1))
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(gcn_model.parameters(), max_norm=1.0)

            optimizer.step()

            if epoch % 5 == 0:
                logger.info("Epoch %03d, Loss: %.6f", epoch, loss.item())

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                # Save best model
                torch.save(gcn_model.state_dict(), "best_gcn_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

            scheduler.step(loss)
        return gcn_model


# Load and prepare data
df_train = load_and_prepare_data(TRAIN_CSV_PATH)
df_test = load_and_prepare_data(TEST_CSV_PATH)
logger.info("Train data: %s", df_train.shape)
logger.info("Test data: %s", df_test.shape)

feature_cols = COMMENT_FEATURE_NAMES + USER_FEATURE_NAMES

# Create datasets with normalization
train_data, scaler = create_pyg_dataset(df_train, feature_cols)
test_data, _ = create_pyg_dataset(df_test, feature_cols, scaler=scaler)

train_data = train_data.to(DEVICE)
test_data = test_data.to(DEVICE)

model = train()

if model is None:
    logger.error("No model, exiting.")
    exit(1)

# Load best model
model.load_state_dict(torch.load("best_gcn_model.pt"))

# Evaluate on test set
predictions, ground_truth, test_mse, test_mae, test_rmse = evaluate_model(
    model, test_data, DEVICE
)

logger.info("=" * 50)
logger.info("TEST SET RESULTS:")
logger.info("Test MSE: %.6f", test_mse)
logger.info("Test MAE: %.6f", test_mae)
logger.info("Test RMSE: %.6f", test_rmse)
logger.info("=" * 50)

# Show some sample predictions
logger.info("\nSample Predictions vs Ground Truth:")
for i in range(min(10, len(predictions))):
    logger.info(f"Pred: {predictions[i]:.4f}, True: {ground_truth[i]:.4f}")

    # Save all predictions and ground truths to CSV
    results_df = pd.DataFrame(
        {
            "prediction": predictions,
            "ground_truth": ground_truth,
            "absolute_error": np.abs(predictions - ground_truth),
        }
    )
    results_df.to_csv("gcn_predictions.csv", index=False)
    logger.info("\nPredictions saved to 'gcn_predictions.csv'")

# Save predictions
results_df = pd.DataFrame(
    {
        "prediction": predictions,
        "ground_truth": ground_truth,
        "absolute_error": np.abs(predictions - ground_truth),
    }
)
results_df.to_csv("gcn_predictions.csv", index=False)
logger.info("\nPredictions saved to 'gcn_predictions.csv'")

plot_results()
