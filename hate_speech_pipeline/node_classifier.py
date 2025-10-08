import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch_geometric.data import Data
from tqdm import tqdm

from hate_speech_pipeline.builder import generate_comment_embeddings
from hate_speech_pipeline.static_models import StaticGCN

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(module)s(%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def load_and_prepare_static_data(
    csv_path: str, subset_count: int, generate_embeddings: bool
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if subset_count > 0:
        df = df.sample(n=subset_count, random_state=42)

    # Drop NAs and filter subreddit without regex overhead
    mask_valid = df["subreddit"].notna() & ~df["subreddit"].str.contains(" ")
    df = df[mask_valid & df["created_utc"].notna()]

    # Vectorized datetime conversion
    df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df[df["timestamp"].dt.year >= 2015]

    # Handle embeddings
    if generate_embeddings:
        df["body_emb"] = list(generate_comment_embeddings(df, "body"))
        df.to_csv(f"{csv_path.replace('.csv', '')}_with_embeddings.csv", index=False)
    else:
        translation = str.maketrans({"[": "", "]": "", "\n": " "})
        cleaned = (
            df["body_emb"]
            .astype(str)
            .str.translate(translation)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df["body_emb"] = cleaned.map(
            lambda s: np.fromstring(s, dtype=np.float32, sep=" ")
        )

    return df


def evaluate_static_model(model, data) -> pd.DataFrame:
    """Get predictions and calculate metrics"""
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index).cpu().numpy().flatten()
        ground_truth = data.y.cpu().numpy().flatten()

        df_val = pd.DataFrame(columns=["mse", "mae", "r2", "log_loss"])
        print_metrics(predictions, ground_truth, df_val)
        return df_val


def print_metrics(all_preds, all_labels, df_results) -> pd.DataFrame:
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    eps = 1e-15

    y_pred_clipped = np.clip(all_preds, eps, 1 - eps)
    ll = -np.mean(
        all_labels * np.log(y_pred_clipped)
        + (1 - all_labels) * np.log(1 - y_pred_clipped)
    )

    if df_results is not None:
        df_results.loc[len(df_results)] = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "log_loss": ll,
        }

    logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    logger.info("MSE = %.4f", mse)
    logger.info("MAE = %.4f", mae)
    logger.info("R2 = %.4f", r2)
    logger.info("Log Loss = %.4f", ll)
    logger.debug("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return df_results


def train_static_gcn(epochs, train_data, device):
    if train_data.x is None:
        return None

    gcn_model = StaticGCN(
        num_node_features=train_data.x.shape[1],
        output_dimension=1,
    ).to(device)

    optimizer = optim.AdamW(gcn_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Binary Cross-Entropy Loss
    loss_fn = nn.BCELoss()

    best_loss = float("inf")
    patience_counter = 0
    patience = 15

    gcn_model.train()
    for epoch in tqdm(range(1, epochs), desc="Training Epochs"):
        optimizer.zero_grad()
        output = gcn_model(train_data.x, train_data.edge_index)
        loss = loss_fn(output.view(-1), train_data.y.view(-1))
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(gcn_model.parameters(), max_norm=1.0)

        optimizer.step()

        logger.debug("Epoch %03d, Loss: %.6f", epoch, loss.item())

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
