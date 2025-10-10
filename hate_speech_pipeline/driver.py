import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from tqdm import tqdm

from hate_speech_pipeline.builder import (
    build_node_mappings,
    build_temporal_graph_local_diffusion,
    load_and_prepare_data,
)
from hate_speech_pipeline.node_classifier import (
    create_static_graph_dataset,
    evaluate_static_model,
    load_and_prepare_static_data,
    train_static_gcn,
)
from model.temporal import BasicRecurrentGCN

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(module)s(%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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
WINDOW_SIZE_HOURS = 2
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def save_predictions_and_print_metrics(
    all_preds, all_labels, df_results, threshold, run_validation=False
):
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
    accuracy = accuracy_score(df_out["ground_truth"], df_out["prediction"])
    precision = precision_score(df_out["ground_truth"], df_out["prediction"])
    f1 = f1_score(df_out["ground_truth"], df_out["prediction"])
    recall = recall_score(df_out["ground_truth"], df_out["prediction"])
    cm = confusion_matrix(df_out["ground_truth"], df_out["prediction"])
    labels = ["Has not propagated hate (0)", "Has propagated hate (1)"]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    markdown_table = cm_df.to_markdown()
    if df_results is not None:
        df_results.loc[len(df_results)] = {
            "threshold": threshold,
            "mse": mse,
            "accuracy": accuracy,
            "precision": precision,
            "f1": f1,
            "recall": recall,
        }

    logger.debug("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    logger.debug("MSE = %.4f", mse)
    logger.debug("Accuracy = %.4f", accuracy)
    logger.debug("Precision = %.4f", precision)
    logger.debug("F1 Score = %.4f", f1)
    logger.debug("Recall = %.4f", recall)
    if not run_validation:
        logger.info("Confusion Matrix:\n%s", markdown_table)
    logger.debug("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # df_out.to_csv(output_path, index=False)
    # logger.info("Saved test predictions to %s", output_path)


def get_graph(
    author2idx, subreddit2idx, num_subreddits, df
) -> DynamicGraphTemporalSignal:
    (
        train_edge_indices,
        train_edge_weights,
        train_features,
        train_labels,
        train_time_vs_count,
        train_masks,
    ) = build_temporal_graph_local_diffusion(
        df=df,
        author2idx=author2idx,
        subreddit2idx=subreddit2idx,
        num_subreddits=num_subreddits,
        comment_feature_names=COMMENT_FEATURE_NAMES,
        user_feature_names=USER_FEATURE_NAMES,
        tox_thresh=0.5,
    )
    dgts = DynamicGraphTemporalSignal(
        edge_indices=train_edge_indices,
        edge_weights=train_edge_weights,
        features=train_features,
        targets=train_labels,
        masks=train_masks,
    )
    logger.info("Built graph with %d snapshots.", dgts.snapshot_count)
    return dgts


def train_model(
    train_dataset: DynamicGraphTemporalSignal,
    epochs=10,
    hidden_dim=128,
    dropout=0.1,
    num_heads=8,
    learning_rate=0.0001,
    val_dataset: DynamicGraphTemporalSignal | None = None,
    patience: int = 10,
    min_delta: float = 1e-4,
    restore_best: bool = True,
):
    model = BasicRecurrentGCN(
        node_features=train_dataset.features[0][0].shape[0],
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_heads=num_heads,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    flat_targets = np.concatenate(train_dataset.targets).ravel()
    num_pos = np.sum(flat_targets == 1)
    num_neg = np.sum(flat_targets == 0)
    logger.info(
        "Number of positive samples: %d, Number of negative samples: %d",
        num_pos,
        num_neg,
    )
    pos_weight = torch.tensor([num_neg / num_pos], device=DEVICE, dtype=torch.float32)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = AsymmetricLoss1(gamma_neg=1, gamma_pos=5, clip=0.05)
    # criterion = FocalLoss(alpha= num_neg / (num_pos + num_neg), gamma=3.0)

    best_state = None
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        epoch_loss = 0.0
        for train_snapshot in train_dataset:
            model.train()
            optimizer.zero_grad()
            output = model(
                train_snapshot.x.to(DEVICE), train_snapshot.edge_index.to(DEVICE)
            )
            loss = criterion(output.view(-1), train_snapshot.y.to(DEVICE).view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / max(1, train_dataset.snapshot_count)
        logger.info(
            "Epoch %d, Average Epoch Loss: %.6f",
            epoch,
            avg_epoch_loss,
        )

        # If a validation dataset is provided, compute validation loss and check early stopping
        if val_dataset is not None:
            model.eval()
            val_loss_accum = 0.0
            val_samples = 0
            with torch.no_grad():
                for val_snapshot_idx, val_snapshot in enumerate(val_dataset):
                    # masks indicate which nodes to evaluate
                    mask = torch.tensor(
                        val_dataset.masks[val_snapshot_idx],
                        dtype=torch.bool,
                        device=DEVICE,
                    )
                    if mask.sum() == 0:
                        continue
                    logits = model(
                        val_snapshot.x.to(DEVICE), val_snapshot.edge_index.to(DEVICE)
                    )
                    masked_logits = logits[mask].view(-1)
                    masked_labels = val_snapshot.y.to(DEVICE)[mask].view(-1)
                    loss_val = criterion(masked_logits, masked_labels)
                    val_loss_accum += loss_val.item() * masked_labels.numel()
                    val_samples += masked_labels.numel()

            val_loss = val_loss_accum / val_samples if val_samples > 0 else float("inf")
            logger.info("Epoch %d, Validation Loss: %.6f", epoch, val_loss)

            # Check improvement
            if val_loss + min_delta < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # save best model weights
                best_state = deepcopy(model.state_dict())
                logger.info(
                    "New best model found at epoch %d with val_loss=%.6f",
                    epoch,
                    val_loss,
                )
            else:
                epochs_no_improve += 1
                logger.debug(
                    "No improvement in val_loss for %d epochs", epochs_no_improve
                )

            if epochs_no_improve >= patience:
                logger.info(
                    "Early stopping triggered (no improvement for %d epochs).", patience
                )
                break

    # Restore best weights if requested
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "early_stopped": (
            epochs_no_improve >= patience if val_dataset is not None else False
        ),
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "stopped_after_epochs_no_improve": epochs_no_improve,
    }

    return model, criterion, info


def evaluate_model(
    model,
    test_dataset: DynamicGraphTemporalSignal,
    criterion,
    threshold=0.2,
    df_results: pd.DataFrame | None = None,
    run_validation=False,
) -> float:
    test_masks = test_dataset.masks
    all_preds = []
    all_labels = []
    model.eval()
    total_samples = 0
    total_loss = 0
    total_samples = 0
    prob_min = 1
    prob_max = -1

    with torch.no_grad():
        for idx, snapshot in tqdm(
            enumerate(test_dataset),
            desc=f"Testing Snapshots at threshold {threshold}",
        ):
            test_mask = torch.tensor(test_masks[idx], dtype=torch.bool, device=DEVICE)
            if test_mask.sum() == 0:
                continue

            logits = model(snapshot.x.to(DEVICE), snapshot.edge_index.to(DEVICE))
            masked_logits = logits[test_mask].view(-1)
            masked_labels = snapshot.y.to(DEVICE)[test_mask].view(-1)

            loss = criterion(masked_logits, masked_labels)
            total_loss += loss.item() * masked_labels.numel()
            total_samples += masked_labels.numel()

            probs = torch.sigmoid(masked_logits).cpu()
            all_labels.append(masked_labels.cpu())
            all_preds.append((probs > threshold).int())

            prob_min = min(prob_min, probs.min().item())
            prob_max = max(prob_max, probs.max().item())

    avg_loss = total_loss / total_samples if total_samples > 0 else float("nan")

    logger.info(
        "Average testing Loss = %.4f at threshold %.2f",
        avg_loss,
        threshold,
    )
    all_preds = torch.cat(all_preds).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()

    save_predictions_and_print_metrics(
        all_preds, all_labels, df_results, threshold, run_validation=run_validation
    )
    logger.info(
        "Prediction Probability Range (Min-Max): [%.3f, %.3f]", prob_min, prob_max
    )
    logger.info(
        "Prediction Value Range (All): [%.3f, %.3f], Mean: %.3f",
        all_preds.min().item(),
        all_preds.max().item(),
        all_preds.mean().item(),
    )
    # Not used in cross validation.
    return (prob_min + prob_max) / 2


def run_diffusion(
    generate_embeddings=False,
    train_file_path="train_dataset_with_emb.csv",
    val_file_path="val_dataset_with_emb.csv",
    test_file_path="test_dataset_with_emb.csv",
    subset_count=0,
    window_size_hours=WINDOW_SIZE_HOURS,
    epochs=10,
):
    return run_diffusion_train_test(
        generate_embeddings=generate_embeddings,
        train_file_path=train_file_path,
        val_file_path=val_file_path,
        test_file_path=test_file_path,
        subset_count=subset_count,
        window_size_hours=window_size_hours,
        epochs=epochs,
    )


def run_diffusion_cv(
    generate_embeddings=False,
    train_file_path="train_dataset_with_emb.csv",
    val_file_path="val_dataset_with_emb.csv",
    test_file_path="test_dataset_with_emb.csv",
    subset_count=0,
    window_size_hours=WINDOW_SIZE_HOURS,
    epochs=10,
    patience: int = 5,
    min_delta: float = 1e-4,
    save_path: str | None = None,
):
    """Cross-validation / grid-logging mode: run grid over hyperparams and log per-config metrics.
    Returns a pandas DataFrame with rows for each config x threshold.
    """
    logger.info("Preparing windows for %s hours.", window_size_hours)
    df_train = load_and_prepare_data(
        train_file_path,
        window_size_hours,
        generate_embeddings=generate_embeddings,
        subset_count=subset_count,
    )
    logger.info("Loaded training data with %d rows.", len(df_train))

    df_test = load_and_prepare_data(
        test_file_path,
        window_size_hours,
        generate_embeddings=generate_embeddings,
        subset_count=subset_count,
    )
    logger.info("Loaded testing data with %d rows.", len(df_test))

    df_val = load_and_prepare_data(
        val_file_path,
        window_size_hours,
        generate_embeddings=generate_embeddings,
        subset_count=subset_count,
    )
    logger.info("Loaded validation data with %d rows.", len(df_val))

    df_combined = pd.concat([df_train, df_test, df_val], ignore_index=True)
    author2idx, subreddit2idx, num_subreddits = build_node_mappings(df_combined)

    train_dataset = get_graph(author2idx, subreddit2idx, num_subreddits, df_train)
    val_dataset = get_graph(author2idx, subreddit2idx, num_subreddits, df_val)

    # Hyperparameter grid search over parameters accepted by train_model
    # (epochs, hidden_dim, dropout, num_heads, learning_rate)
    epochs_list = [max(1, epochs // 2), epochs] if epochs > 1 else [epochs]
    hidden_dims = [64, 128]
    dropouts = [0.1, 0.3]
    num_heads_list = [4, 8]
    learning_rates = [1e-3, 1e-4]

    logger.info("Starting hyperparameter logging over DCRNN grid...")

    # Collect per-config-per-threshold results here
    grid_results = []

    for ep in epochs_list:
        for hd in hidden_dims:
            for do in dropouts:
                for nh in num_heads_list:
                    for lr in learning_rates:
                        logger.info(
                            "Training config: epochs=%d, hidden_dim=%d, dropout=%.2f, num_heads=%d, lr=%.4g",
                            ep,
                            hd,
                            do,
                            nh,
                            lr,
                        )

                        model, crit, info = train_model(
                            train_dataset,
                            epochs=ep,
                            hidden_dim=hd,
                            dropout=do,
                            num_heads=nh,
                            learning_rate=lr,
                            val_dataset=val_dataset,
                            patience=patience,
                            min_delta=min_delta,
                            restore_best=True,
                        )

                        # persist best model if early stopped and a save_path provided
                        if info.get("early_stopped") and save_path:
                            try:
                                torch.save(model.state_dict(), save_path)
                                logger.info(
                                    "Saved early-stopped model to %s", save_path
                                )
                            except Exception as e:
                                logger.error(
                                    "Failed to save model to %s: %s", save_path, e
                                )

                        # Evaluate this model on the validation set across thresholds
                        df_val_config = pd.DataFrame(
                            columns=[
                                "threshold",
                                "mse",
                                "accuracy",
                                "precision",
                                "recall",
                                "f1",
                            ]
                        )
                        for threshold_candidate in range(20, 41, 2):
                            evaluate_model(
                                model,
                                val_dataset,
                                crit,
                                threshold=threshold_candidate / 100,
                                df_results=df_val_config,
                                run_validation=True,
                            )

                        # Append each threshold row to grid_results with hyperparams
                        if df_val_config.empty:
                            logger.warning(
                                "No validation results for config epochs=%s, hidden_dim=%s, dropout=%s, num_heads=%s, lr=%s",
                                ep,
                                hd,
                                do,
                                nh,
                                lr,
                            )
                        else:
                            for _, row in df_val_config.iterrows():
                                grid_results.append(
                                    {
                                        "epochs": ep,
                                        "hidden_dim": hd,
                                        "dropout": do,
                                        "num_heads": nh,
                                        "learning_rate": lr,
                                        "threshold": row["threshold"],
                                        "mse": row["mse"],
                                        "accuracy": row["accuracy"],
                                        "precision": row["precision"],
                                        "recall": row["recall"],
                                        "f1": row["f1"],
                                    }
                                )

                        # free intermediate model to avoid holding memory
                        try:
                            del model
                            del crit
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass

    # Create a single DataFrame with all grid results and return it
    df_grid_results = pd.DataFrame(grid_results)
    if df_grid_results.empty:
        logger.warning("Grid search produced no results.")
    else:
        logger.info(
            "Grid search completed. Results:\n%s", df_grid_results.to_markdown()
        )
        # Optionally persist
        # df_grid_results.to_csv("dcrnn_grid_results.csv", index=False)

    return df_grid_results


def run_diffusion_train_test(
    generate_embeddings=False,
    train_file_path="train_dataset_with_emb.csv",
    val_file_path="val_dataset_with_emb.csv",
    test_file_path="test_dataset_with_emb.csv",
    subset_count=0,
    window_size_hours=WINDOW_SIZE_HOURS,
    epochs=10,
    patience: int = 5,
    min_delta: float = 1e-4,
    save_path: str | None = "best_dcrnn_model.pt",
):
    """Plain train/test: trains a single model on train set and evaluates on validation to pick threshold, then evaluates on test set."""
    logger.info("Preparing windows for %s hours.", window_size_hours)
    df_train = load_and_prepare_data(
        train_file_path,
        window_size_hours,
        generate_embeddings=generate_embeddings,
        subset_count=subset_count,
    )
    logger.info("Loaded training data with %d rows.", len(df_train))
    df_test = load_and_prepare_data(
        test_file_path,
        window_size_hours,
        generate_embeddings=generate_embeddings,
        subset_count=subset_count,
    )
    logger.info("Loaded testing data with %d rows.", len(df_test))

    df_val = load_and_prepare_data(
        val_file_path,
        window_size_hours,
        generate_embeddings=generate_embeddings,
        subset_count=subset_count,
    )
    logger.info("Loaded validation data with %d rows.", len(df_val))

    df_combined = pd.concat([df_train, df_test, df_val], ignore_index=True)
    author2idx, subreddit2idx, num_subreddits = build_node_mappings(df_combined)

    train_dataset = get_graph(author2idx, subreddit2idx, num_subreddits, df_train)
    val_dataset = get_graph(author2idx, subreddit2idx, num_subreddits, df_val)
    test_dataset = get_graph(author2idx, subreddit2idx, num_subreddits, df_test)

    df_val = pd.DataFrame(
        columns=["threshold", "mse", "accuracy", "precision", "recall", "f1_score"]
    )
    df_results = df_val.copy()

    # Extract model / evaluation parameters into variables

    train_hidden_dim = 128
    train_dropout = 0.1
    train_num_heads = 4
    train_learning_rate = 0.0001

    logger.info(
        "Training with fixed params: epochs=%d, hidden_dim=%d, dropout=%.2f, num_heads=%d, lr=%.4g",
        epochs,
        train_hidden_dim,
        train_dropout,
        train_num_heads,
        train_learning_rate,
    )

    dcrnn_model, criterion, info = train_model(
        train_dataset,
        epochs=epochs,
        hidden_dim=train_hidden_dim,
        dropout=train_dropout,
        num_heads=train_num_heads,
        learning_rate=train_learning_rate,
        val_dataset=val_dataset,
        patience=patience,
        min_delta=min_delta,
        restore_best=True,
    )

    # Persist model if early stopping occurred and save_path provided
    if info.get("early_stopped") and save_path:
        try:
            torch.save(dcrnn_model.state_dict(), save_path)
            logger.info("Saved early-stopped model to %s", save_path)
        except Exception as e:
            logger.error("Failed to save model to %s: %s", save_path, e)

    # Use provided threshold for validation and test evaluation
    provided_threshold = 0.4
    # thresholds sweep variables (percent integer values)
    val_thresh_start = 1
    val_thresh_end = 50
    val_thresh_step = 3
    avg_probs = []

    for threshold_candidate in range(val_thresh_start, val_thresh_end, val_thresh_step):
        avg_probs.append(
            evaluate_model(
                dcrnn_model,
                val_dataset,
                criterion,
                threshold=threshold_candidate / 100,
                df_results=df_val,
                run_validation=True,
            )
        )

    logger.info("Validation evaluations completed. Results:\n%s", df_val.to_markdown())

    # Prefer the provided_threshold for final evaluation, but attempt to choose from validation
    best_threshold = provided_threshold
    try:
        chosen = (
            df_val[abs(df_val["accuracy"] - df_val["recall"]) <= 0.5]
            .sort_values("recall", ascending=False)
            .head(1)["threshold"]
            .values[0]
        )
        if chosen is not None:
            best_threshold = chosen
    except Exception as e:
        logger.debug(
            "Could not compute chosen threshold from validation results: %s", e
        )
        logger.info("Average probs: %s", avg_probs)
        logger.info("Backup threshold = %.2f", np.mean(avg_probs) if avg_probs else 0)
        best_threshold = np.mean(avg_probs) if avg_probs else provided_threshold

    logger.info("Using threshold %.2f for final test evaluation.", best_threshold)
    evaluate_model(
        dcrnn_model,
        test_dataset,
        criterion,
        threshold=best_threshold,
        df_results=df_results,
    )
    logger.info("Completed all evaluations. Results:\n%s", df_results.to_markdown())

    return df_results


def run_classification(
    generate_embeddings=False,
    train_file_path="train_dataset_with_emb.csv",
    val_file_path="val_dataset_with_emb.csv",
    test_file_path="test_dataset_with_emb.csv",
    subset_count=0,
    epochs=10,
):
    # Load and prepare data
    df_train = load_and_prepare_static_data(
        train_file_path, subset_count, generate_embeddings
    )
    df_test = load_and_prepare_static_data(
        test_file_path, subset_count, generate_embeddings
    )
    df_val = load_and_prepare_static_data(
        val_file_path, subset_count, generate_embeddings
    )
    logger.info("Train data: %s", df_train.shape)
    logger.info("Test data: %s", df_test.shape)

    feature_cols = COMMENT_FEATURE_NAMES + USER_FEATURE_NAMES

    # Create datasets with normalization
    train_data, scaler = create_static_graph_dataset(df_train, feature_cols)
    test_data, _ = create_static_graph_dataset(df_test, feature_cols, scaler=scaler)
    val_data, _ = create_static_graph_dataset(df_val, feature_cols, scaler=scaler)

    train_data = train_data.to(DEVICE)
    test_data = test_data.to(DEVICE)
    val_data = val_data.to(DEVICE)

    # Hyperparameter grid
    hidden_dims = [196, 256]
    learning_rates = [0.001, 0.0005]
    decays = [1e-5, 1e-4]
    dropouts = [0.1, 0.3]

    cv_results = []
    for hidden_dim in tqdm(hidden_dims, desc="Hidden Dims"):
        for learning_rate in tqdm(learning_rates, desc="Learning Rates"):
            for decay in tqdm(decays, desc="Decays"):
                for dropout in tqdm(dropouts, desc="Dropouts"):
                    logger.info(
                        "CV: hidden_dim=%d, lr=%.5f, decay=%.1e, dropout=%.2f",
                        hidden_dim,
                        learning_rate,
                        decay,
                        dropout,
                    )
                    model = train_static_gcn(
                        epochs,
                        train_data,
                        device=DEVICE,
                        hidden_dim=hidden_dim,
                        learning_rate=learning_rate,
                        decay=decay,
                        dropout=dropout,
                    )
                    if model is None:
                        logger.error("No model, skipping this config.")
                        continue
                    model.load_state_dict(torch.load("best_gcn_model.pt"))
                    df_results = evaluate_static_model(model, val_data, cv=True)
                    logger.debug(
                        "Test set evaluation results:\n%s", df_results.to_markdown()
                    )
                    cv_results.append(
                        {
                            "hidden_dim": hidden_dim,
                            "learning_rate": learning_rate,
                            "decay": decay,
                            "dropout": dropout,
                            **df_results.iloc[0].to_dict(),
                        }
                    )

    if cv_results:
        df_cv = pd.DataFrame(cv_results)
        logger.info("\nCV summary:\n%s", df_cv.to_markdown())
        if "r2" in df_cv.columns:
            best_idx = df_cv["r2"].idxmax()
            best_row = df_cv.loc[best_idx]
            if isinstance(best_row, pd.DataFrame):
                best_row = best_row.iloc[0]
            logger.info(
                "Best r2: %.4f with params: hidden_dim=%d, learning_rate=%.5f, decay=%.1e, dropout=%.2f",
                best_row["r2"],
                best_row["hidden_dim"],
                best_row["learning_rate"],
                best_row["decay"],
                best_row["dropout"],
            )

            # Retrain on full train+val and test on test_data
            logger.info("Retraining on best params.")
            full_train_df = pd.concat([df_train, df_val], ignore_index=True)
            full_train_data, scaler = create_static_graph_dataset(
                full_train_df, feature_cols
            )
            test_data, _ = create_static_graph_dataset(
                df_test, feature_cols, scaler=scaler
            )
            full_train_data = full_train_data.to(DEVICE)
            test_data = test_data.to(DEVICE)
            best_model = train_static_gcn(
                epochs,
                full_train_data,
                device=DEVICE,
                hidden_dim=int(
                    getattr(
                        best_row["hidden_dim"], "item", lambda: best_row["hidden_dim"]
                    )()
                ),
                learning_rate=float(
                    getattr(
                        best_row["learning_rate"],
                        "item",
                        lambda: best_row["learning_rate"],
                    )()
                ),
                decay=float(
                    getattr(best_row["decay"], "item", lambda: best_row["decay"])()
                ),
                dropout=float(
                    getattr(best_row["dropout"], "item", lambda: best_row["dropout"])()
                ),
            )
            if best_model is not None:
                best_model.load_state_dict(torch.load("best_gcn_model.pt"))
                test_results = evaluate_static_model(best_model, test_data, cv=False)
                logger.info(
                    "Test set results with best params:\n%s", test_results.to_markdown()
                )

    # results_df.to_csv("gcn_predictions.csv", index=False)
    # logger.info("\nPredictions saved to 'gcn_predictions.csv'")


def run(
    generate_embeddings=False,
    train_file_path="train_dataset_with_emb.csv",
    val_file_path="val_dataset_with_emb.csv",
    test_file_path="test_dataset_with_emb.csv",
    subset_count=0,
    window_size_hours=WINDOW_SIZE_HOURS,
    epochs=10,
    mode="diffusion",
):
    if mode == "diffusion":
        run_diffusion(
            generate_embeddings=generate_embeddings,
            train_file_path=train_file_path,
            val_file_path=val_file_path,
            test_file_path=test_file_path,
            subset_count=subset_count,
            window_size_hours=window_size_hours,
            epochs=epochs,
        )
    else:
        run_classification(
            generate_embeddings=generate_embeddings,
            train_file_path=train_file_path,
            val_file_path=val_file_path,
            test_file_path=test_file_path,
            subset_count=subset_count,
            epochs=epochs,
        )
