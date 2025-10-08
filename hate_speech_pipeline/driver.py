import logging
import os

import numpy as np
import pandas as pd
import torch

# 方法2: Set PyTorch memory management for large datasets
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from torch.cuda.amp import autocast, GradScaler
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from tqdm import tqdm

from hate_speech_pipeline.builder import (
    build_node_mappings,
    build_temporal_graph_local_diffusion,
    load_and_prepare_data,
)
from hate_speech_pipeline.temporal_models import BasicRecurrentGCN
from hate_speech_pipeline.loss_functions import FocalLoss
from imblearn.over_sampling import SMOTE

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
    "user_avg_posting_interval",
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
    precision = precision_score(df_out["ground_truth"], df_out["prediction"], zero_division=0)
    f1 = f1_score(df_out["ground_truth"], df_out["prediction"], zero_division=0)
    recall = recall_score(df_out["ground_truth"], df_out["prediction"], zero_division=0)
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
            "f1_score": f1,
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
    logger.info("Built graph with %d snapshots.", len(train_edge_indices))
    return DynamicGraphTemporalSignal(
        edge_indices=train_edge_indices,
        edge_weights=train_edge_weights,
        features=train_features,
        targets=train_labels,
        masks=train_masks,
    )


def train_model(
    train_dataset: DynamicGraphTemporalSignal,
    epochs=10,
    hidden_dim=128,
    dropout=0.1,
    num_heads=8,
    learning_rate=0.0001,
    accumulation_steps=1,  # 不使用梯度累积    
):
    # 方法3: Clear GPU cache before training
    torch.cuda.empty_cache()
    
    model = BasicRecurrentGCN(
        node_features=train_dataset.features[0][0].shape[0],
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_heads=num_heads,
    ).to(DEVICE)
    
    # 调整学习率（因为有效批次大小增加了），添加权重衰减
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate * accumulation_steps, weight_decay=1e-4)
    
    # 增加梯度累积步数以进一步减少内存使用
    accumulation_steps = min(accumulation_steps, 4)  # 限制最大累积步数
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    # 创建梯度缩放器 (混合精度训练)
    scaler = GradScaler()
    flat_targets = np.concatenate(train_dataset.targets).ravel()
    num_pos = np.sum(flat_targets == 1)
    num_neg = np.sum(flat_targets == 0)
    logger.info(
        "Number of positive samples: %d, Number of negative samples: %d",
        num_pos,
        num_neg,
    )
    pos_weight = torch.tensor([num_neg / num_pos], device=DEVICE, dtype=torch.float32)

    # 使用BCE Loss with pos_weight来处理类别不平衡
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        epoch_loss = 0
        optimizer.zero_grad()
        
        for i, train_snapshot in enumerate(train_dataset):
            model.train()
            
            # 使用混合精度训练
            with autocast():
                output = model(
                    train_snapshot.x.to(DEVICE), train_snapshot.edge_index.to(DEVICE)
                )
                loss = criterion(output.view(-1), train_snapshot.y.to(DEVICE).view(-1))
                
                # 缩放损失（除以累积步数）
                loss = loss / accumulation_steps
            
            # 缩放损失并反向传播
            scaler.scale(loss).backward()
            
            # 每accumulation_steps步更新一次参数
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps

        # 更新学习率
        scheduler.step()
        
        logger.debug(
            "Epoch %d, Average Epoch Loss: %.6f, Learning Rate: %.6f",
            epoch,
            epoch_loss / train_dataset.snapshot_count,
            scheduler.get_last_lr()[0]
        )
        
        # 方法3: Clear GPU cache after each epoch
        torch.cuda.empty_cache()
        
    return model, criterion


def evaluate_model(
    model,
    test_dataset: DynamicGraphTemporalSignal,
    criterion,
    threshold=0.2,
    df_results: pd.DataFrame | None = None,
    run_validation=False,
):
    test_masks = test_dataset.masks
    all_preds = []
    all_labels = []
    model.eval()
    total_samples = 0
    total_loss = 0
    total_samples = 0
    prob_min = 1.0
    prob_max = 0.0

    with torch.no_grad():
        for idx, snapshot in tqdm(
            enumerate(test_dataset),
            desc=f"Testing Snapshots at threshold {threshold}",
        ):
            test_mask = torch.tensor(test_masks[idx], dtype=torch.bool, device=DEVICE)
            if test_mask.sum() == 0:
                continue

            with autocast():
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


def run(
    generate_embeddings=False,
    train_file_path="train_dataset_with_emb.csv",
    val_file_path="val_dataset_with_emb.csv",
    test_file_path="test_dataset_with_emb.csv",
    subset_count=0,
    window_size_hours=WINDOW_SIZE_HOURS,
    epochs=10,
):
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
    dcrnn_model, criterion = train_model(train_dataset, epochs=epochs)

    for threshold_candidate in range(20, 61, 4):
        evaluate_model(
            dcrnn_model,
            val_dataset,
            criterion,
            threshold=threshold_candidate / 100,
            df_results=df_val,
            run_validation=True,
        )

    logger.info("Validation evaluations completed. Results:\n%s", df_val.to_markdown())
    
    # 修复空结果的问题
    filtered_df = df_val[abs(df_val["accuracy"] - df_val["recall"]) <= 0.5]
    if len(filtered_df) > 0:
        best_threshold = (
            filtered_df.sort_values("recall", ascending=False)
            .head(1)["threshold"]
            .values[0]
        )
        logger.info("Best threshold selected: %.2f", best_threshold)
    else:
        # 如果没有合适的阈值，选择F1分数最高的
        if len(df_val) > 0:
            best_threshold = df_val.sort_values("f1_score", ascending=False, na_position='last').head(1)["threshold"].values[0]
            logger.info("No optimal threshold found, using best F1 threshold: %.2f", best_threshold)
        else:
            best_threshold = 0.5  # 默认阈值
            logger.info("No validation results, using default threshold: %.2f", best_threshold)
    evaluate_model(
        dcrnn_model,
        test_dataset,
        criterion,
        threshold=best_threshold,
        df_results=df_results,
    )
    logger.info("Completed all evaluations. Results:\n%s", df_results.to_markdown())
