import logging
import os
from typing import Dict, List, Tuple, Any
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from hate_speech_pipeline.driver import get_graph, evaluate_model
from hate_speech_pipeline.temporal_models import BasicRecurrentGCN
from hate_speech_pipeline.builder import build_node_mappings, load_and_prepare_data

# Set memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(module)s(%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class HyperparameterTuner:
    """超参数微调器"""
    
    def __init__(self, output_dir="tuning_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        
    def get_hyperparameter_grid(self) -> Dict[str, List]:
        """定义超参数搜索空间"""
        return {
            # 模型参数
            "hidden_dim": [64, 96, 128, 256],
            "num_heads": [2, 4, 6, 8],
            "dropout": [0.1, 0.2, 0.3, 0.5],
            
            # 训练参数
            "learning_rate": [0.001, 0.0005, 0.0001, 0.00005],
            "epochs": [10, 20, 50],
            "optimizer": ["adam", "adamw"],
            "weight_decay": [0, 1e-5, 1e-4, 1e-3],  # 新增：L2正则化
            
            # 损失函数参数
            "loss_type": ["bce", "focal"],  # 新增：损失函数类型
            "pos_weight_multiplier": [1.0, 1.5, 2.0, 3.0],  # 新增：正样本权重倍数
            "focal_alpha": [0.25, 0.5, 0.75],  # 新增：Focal Loss alpha
            "focal_gamma": [1.0, 2.0, 3.0],  # 新增：Focal Loss gamma
            
            # 数据参数
            "window_size_hours": [1, 2, 4],
            
            # 阈值参数
            "threshold_start": [0.1, 0.2, 0.3, 0.5],  # 扩展阈值搜索范围
            "threshold_end": [0.5, 0.6, 0.7, 0.8],
            "threshold_step": [0.02, 0.05],
        }
    
    def train_single_config(
        self,
        train_dataset,
        val_dataset,
        config: Dict[str, Any],
        trial_id: int
    ) -> Dict[str, Any]:
        """训练单个配置"""
        logger.info(f"Trial {trial_id}: Testing config {config}")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        try:
            # 创建模型
            model = BasicRecurrentGCN(
                node_features=train_dataset.features[0][0].shape[0],
                hidden_dim=config["hidden_dim"],
                dropout=config["dropout"],
                num_heads=config["num_heads"],
            ).to(DEVICE)
            
            # 创建优化器
            if config["optimizer"] == "adam":
                optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0))
            elif config["optimizer"] == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0))
            else:
                optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0))
            
            # 学习率调度器
            scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
            
            # 损失函数
            flat_targets = np.concatenate(train_dataset.targets).ravel()
            num_pos = np.sum(flat_targets == 1)
            num_neg = np.sum(flat_targets == 0)
            
            # 创建损失函数（支持BCE和Focal Loss）
            if config.get("loss_type", "bce") == "focal":
                from hate_speech_pipeline.loss_functions import FocalLoss
                criterion = FocalLoss(
                    alpha=config.get("focal_alpha", 0.25),
                    gamma=config.get("focal_gamma", 2.0)
                )
            else:
                pos_weight = torch.tensor(
                    [num_neg / num_pos * config.get("pos_weight_multiplier", 1.0)],
                    device=DEVICE,
                    dtype=torch.float32
                )
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            # 训练循环
            best_val_f1 = 0.0
            best_metrics = {}
            
            for epoch in range(config["epochs"]):
                model.train()
                epoch_loss = 0
                
                for train_snapshot in train_dataset:
                    optimizer.zero_grad()
                    output = model(
                        train_snapshot.x.to(DEVICE), 
                        train_snapshot.edge_index.to(DEVICE)
                    )
                    loss = criterion(output.view(-1), train_snapshot.y.to(DEVICE).view(-1))
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                scheduler.step()
                torch.cuda.empty_cache()
                
                # 每5个epoch验证一次
                if (epoch + 1) % 5 == 0:
                    val_metrics = self.evaluate_config(model, val_dataset, config)
                    if val_metrics["f1_score"] > best_val_f1:
                        best_val_f1 = val_metrics["f1_score"]
                        best_metrics = val_metrics.copy()
            
            # 添加配置信息到结果
            best_metrics.update(config)
            best_metrics["trial_id"] = trial_id
            best_metrics["best_val_f1"] = best_val_f1
            
            logger.info(f"Trial {trial_id} completed. Best F1: {best_val_f1:.4f}")
            return best_metrics
            
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {str(e)}")
            return {
                "trial_id": trial_id,
                "error": str(e),
                **config
            }
    
    def evaluate_config(
        self, 
        model, 
        val_dataset, 
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """评估单个配置"""
        model.eval()
        all_preds = []
        all_labels = []
        
        # 测试多个阈值
        threshold_start = config["threshold_start"]
        threshold_end = config["threshold_end"]
        threshold_step = config["threshold_step"]
        
        best_f1 = 0.0
        best_metrics = {}
        
        for threshold in np.arange(threshold_start, threshold_end, threshold_step):
            preds = []
            labels = []
            
            with torch.no_grad():
                for idx, snapshot in enumerate(val_dataset):
                    val_mask = torch.tensor(val_dataset.masks[idx], dtype=torch.bool, device=DEVICE)
                    if val_mask.sum() == 0:
                        continue
                    
                    logits = model(snapshot.x.to(DEVICE), snapshot.edge_index.to(DEVICE))
                    masked_logits = logits[val_mask].view(-1)
                    masked_labels = snapshot.y.to(DEVICE)[val_mask].view(-1)
                    
                    probs = torch.sigmoid(masked_logits).cpu()
                    preds.extend((probs > threshold).int().tolist())
                    labels.extend(masked_labels.cpu().int().tolist())
            
            if len(preds) > 0:
                f1 = f1_score(labels, preds, zero_division=0)
                accuracy = accuracy_score(labels, preds)
                precision = precision_score(labels, preds, zero_division=0)
                recall = recall_score(labels, preds, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = {
                        "f1_score": f1,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "threshold": threshold
                    }
        
        return best_metrics
    
    def run_tuning(
        self,
        train_file_path: str,
        val_file_path: str,
        test_file_path: str,
        max_trials: int = 50,
        generate_embeddings: bool = False
    ):
        """运行超参数微调"""
        logger.info("Starting hyperparameter tuning...")
        
        # 加载数据
        logger.info("Loading data...")
        df_train = load_and_prepare_data(
            train_file_path, 1, generate_embeddings=generate_embeddings, subset_count=10000
        )
        df_val = load_and_prepare_data(
            val_file_path, 1, generate_embeddings=generate_embeddings, subset_count=5000
        )
        df_test = load_and_prepare_data(
            test_file_path, 1, generate_embeddings=generate_embeddings, subset_count=5000
        )
        
        # 构建图数据
        df_combined = pd.concat([df_train, df_val, df_test], ignore_index=True)
        author2idx, subreddit2idx, num_subreddits = build_node_mappings(df_combined)
        
        train_dataset = get_graph(author2idx, subreddit2idx, num_subreddits, df_train)
        val_dataset = get_graph(author2idx, subreddit2idx, num_subreddits, df_val)
        
        # 生成超参数组合
        param_grid = self.get_hyperparameter_grid()
        param_combinations = list(ParameterGrid(param_grid))
        
        # 随机采样或全部测试
        if len(param_combinations) > max_trials:
            import random
            param_combinations = random.sample(param_combinations, max_trials)
        
        logger.info(f"Testing {len(param_combinations)} configurations...")
        
        # 运行微调
        for i, config in enumerate(param_combinations):
            result = self.train_single_config(train_dataset, val_dataset, config, i)
            self.results.append(result)
            
            # 每10个试验保存一次结果
            if (i + 1) % 10 == 0:
                self.save_results(f"tuning_progress_{i+1}.json")
        
        # 保存最终结果
        self.save_results("final_tuning_results.json")
        self.analyze_results()
    
    def save_results(self, filename: str):
        """保存微调结果"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {filepath}")
    
    def analyze_results(self):
        """分析微调结果"""
        if not self.results:
            logger.warning("No results to analyze")
            return
        
        # 过滤成功的结果
        successful_results = [r for r in self.results if "error" not in r]
        
        if not successful_results:
            logger.warning("No successful trials to analyze")
            return
        
        # 按F1分数排序
        successful_results.sort(key=lambda x: x.get("best_val_f1", 0), reverse=True)
        
        logger.info("=== TOP 5 CONFIGURATIONS ===")
        for i, result in enumerate(successful_results[:5]):
            logger.info(f"Rank {i+1}: F1={result.get('best_val_f1', 0):.4f}, "
                       f"Config={result}")
        
        # 保存最佳配置
        best_config = successful_results[0]
        best_config_path = os.path.join(self.output_dir, "best_config.json")
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        logger.info(f"Best configuration saved to {best_config_path}")


def main():
    """主函数"""
    tuner = HyperparameterTuner()
    
    tuner.run_tuning(
        train_file_path="retrain_train80.csv",
        val_file_path="retrain_validation10.csv", 
        test_file_path="retrain_test10.csv",
        max_trials=20,  # 可以根据需要调整
        generate_embeddings=False  # 如果已经有嵌入向量就设为False
    )


if __name__ == "__main__":
    main()
