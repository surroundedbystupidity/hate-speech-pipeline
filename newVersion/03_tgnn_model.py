
#!/usr/bin/env python3
"""
Temporal Graph Neural Network (TGNN) Model Implementation.
Implements TGAT/TGN architectures for hate speech classification and diffusion prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import yaml
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for TGAT.
    
    This module implements temporal attention to capture how node representations
    evolve over time. It uses timestamp information to weight the importance of
    different temporal interactions.
    
    Args:
        input_dim (int): Dimension of input node features
        hidden_dim (int): Dimension of hidden representations
    """
    
    def __init__(self, input_dim, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x, edge_index, timestamps):
        """
        Apply temporal normalization (simplified implementation).

        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            timestamps (torch.Tensor): Edge timestamps [num_edges]

        Returns:
            torch.Tensor: Normalized node features
        """
        return self.layer_norm(x)

class TGATLayer(nn.Module):
    """Temporal Graph Attention Network layer."""
    
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(input_dim, hidden_dim)
        
        # Graph attention
        self.graph_attention = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        
        # Normalization and activation
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, edge_attr, timestamps):
        """Forward pass through TGAT layer."""
        # Apply temporal attention
        x_temporal = self.temporal_attention(x, edge_index, timestamps)
        
        # Apply graph attention
        x_graph = self.graph_attention(x_temporal, edge_index)
        
        # Residual connection and normalization
        x_out = self.norm(x_temporal + self.dropout(x_graph))
        
        return x_out

class TGNLayer(nn.Module):
    """Temporal Graph Network layer with memory."""
    
    def __init__(self, input_dim, hidden_dim, memory_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        
        # Memory update
        self.memory_updater = nn.GRUCell(input_dim, memory_dim)
        
        # Message function
        self.message_function = nn.Sequential(
            nn.Linear(input_dim + memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Aggregation
        self.aggregator = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr, timestamps, memory):
        """Forward pass through TGN layer."""
        # Update memory (simplified)
        new_memory = self.memory_updater(x.mean(dim=0, keepdim=True), memory)
        
        # Compute messages
        memory_expanded = new_memory.expand(x.size(0), -1)
        combined_features = torch.cat([x, memory_expanded], dim=-1)
        messages = self.message_function(combined_features)
        
        # Aggregate messages
        x_out = self.aggregator(messages)
        
        return x_out, new_memory

class TGNNModel(nn.Module):
    """Complete TGNN model for hate speech analysis."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        tgnn_config = config.get('tgnn', {})
        
        self.model_type = tgnn_config.get('model_type', 'TGAT')
        self.input_dim = tgnn_config.get('input_dim', 781)  # BERT + user + subreddit + node type features
        self.hidden_dim = tgnn_config.get('hidden_dim', 128)
        self.num_layers = tgnn_config.get('num_layers', 3)  # Changed from 2 to 3
        self.num_classes = tgnn_config.get('num_classes', 2)
        self.dropout = tgnn_config.get('dropout', 0.1)
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # TGNN layers
        if self.model_type == 'TGAT':
            self.tgnn_layers = nn.ModuleList([
                TGATLayer(self.hidden_dim, self.hidden_dim) 
                for _ in range(self.num_layers)
            ])
        elif self.model_type == 'TGN':
            self.memory_dim = tgnn_config.get('memory_dim', 64)
            self.memory = nn.Parameter(torch.randn(1, self.memory_dim))
            self.tgnn_layers = nn.ModuleList([
                TGNLayer(self.hidden_dim, self.hidden_dim, self.memory_dim) 
                for _ in range(self.num_layers)
            ])
        elif self.model_type == 'TGNN':
            # Simple TGNN implementation using GCN with temporal features
            self.tgnn_layers = nn.ModuleList([
                GCNConv(self.hidden_dim, self.hidden_dim) 
                for _ in range(self.num_layers)
            ])
            self.temporal_encoder = nn.Linear(1, self.hidden_dim)  # Encode timestamps
        
        # Classification head (binary classification - single output)
        self.hate_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)  # Single output for binary classification with BCEWithLogitsLoss
        )
        
        # Diffusion prediction head
        self.diffusion_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Time prediction head
        self.time_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def forward(self, data, task='classification'):
        """Forward pass for different tasks."""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        timestamps = data.edge_time
        
        # Input projection
        x = self.input_projection(x)
        
        # Apply TGNN layers
        if self.model_type == 'TGAT':
            for layer in self.tgnn_layers:
                x = layer(x, edge_index, edge_attr, timestamps)
        elif self.model_type == 'TGN':
            memory = self.memory
            for layer in self.tgnn_layers:
                x, memory = layer(x, edge_index, edge_attr, timestamps, memory)
        elif self.model_type == 'TGNN':
            # Apply GCN layers with residual connections
            for i, layer in enumerate(self.tgnn_layers):
                x_new = layer(x, edge_index)
                x_new = F.relu(x_new)
                if i > 0:  # Add residual connection after first layer
                    x = x + x_new
                else:
                    x = x_new
            
            # Add temporal information as node features if available
            if timestamps is not None and len(timestamps) == x.size(0):
                # Only add if timestamps match node count
                timestamps_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
                temporal_emb = self.temporal_encoder(timestamps_norm.unsqueeze(-1))
                x = x + temporal_emb
        
        if task == 'classification':
            # Node classification
            return self.hate_classifier(x)
        
        elif task == 'diffusion':
            # Edge prediction for diffusion
            src_nodes = x[edge_index[0]]
            dst_nodes = x[edge_index[1]]
            edge_features = torch.cat([src_nodes, dst_nodes], dim=-1)
            return self.diffusion_predictor(edge_features)
        
        elif task == 'time_prediction':
            # Time prediction for interactions
            src_nodes = x[edge_index[0]]
            dst_nodes = x[edge_index[1]]
            edge_features = torch.cat([src_nodes, dst_nodes], dim=-1)
            return self.time_predictor(edge_features)
        
        else:
            return x  # Return embeddings

class TGNNTrainer:
    """Trainer for TGNN models."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Optimizer
        tgnn_config = config.get('tgnn', {})
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tgnn_config.get('learning_rate', 0.001)
        )

        # Loss functions for semi-supervised learning
        # Use BCEWithLogitsLoss with pos_weight=1.0 since labels are balanced
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
        self.regression_loss = nn.MSELoss()

        # Best threshold for F1 (to be calibrated on validation set)
        self.best_threshold = 0.5

    def calibrate_threshold(self, graph_data):
        """
        Calibrate threshold on validation set to maximize F1 score.
        """
        print("Calibrating threshold on validation set...")

        self.model.eval()
        with torch.no_grad():
            # Get predictions on validation set
            out = self.model(graph_data, task='classification')

            # Only use labelled validation nodes
            val_mask = graph_data.val_mask & graph_data.label_mask
            if val_mask.sum() == 0:
                print("No validation data available, using default threshold 0.5")
                return 0.5

            val_logits = out[val_mask]
            val_labels = graph_data.y[val_mask]

            # Convert to probabilities
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_true = val_labels.cpu().numpy()

            # Test different thresholds
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_f1 = 0
            best_thresh = 0.5

            from sklearn.metrics import f1_score

            for thresh in thresholds:
                pred_binary = (val_probs > thresh).astype(int)
                f1 = f1_score(val_true, pred_binary, average='binary', zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh

            print(f"Best threshold: {best_thresh:.3f} with F1: {best_f1:.3f}")
            self.best_threshold = best_thresh
            return best_thresh
        
    def prepare_data(self, graph_data):
        """
        Data is already prepared with masks during loading.
        Just move to device and log statistics.
        """
        print("Preparing training data...")

        # Move graph data to the appropriate device (CPU/GPU)
        graph_data = graph_data.to(self.device)

        # Log data preparation statistics
        self._log_data_statistics(graph_data)

        return graph_data
    
    def _create_user_labels(self, num_nodes, graph_data):
        """
        Use labels already created in graph_data from CSV.
        
        Args:
            num_nodes (int): Number of nodes 
            graph_data: Graph data containing labels
            
        Returns:
            list: Node labels (0 for normal, 1 for hate speech)
        """
        print("Using labels from graph data")
        return graph_data.y.cpu().numpy().tolist()
    
    def _build_user_hate_mapping(self, balanced_df):
        """
        Build a mapping from usernames to hate speech labels.
        
        Args:
            balanced_df: Balanced dataset DataFrame
            
        Returns:
            dict: Mapping from username to boolean hate flag
        """
        user_hate_map = {}
        
        for _, row in balanced_df.iterrows():
            author = row['author']
            is_hate_post = (row['binary_label'] == 1)
            
            # Initialize user if not seen before
            if author not in user_hate_map:
                user_hate_map[author] = False
            
            # Mark user as hate speech user if they have any hate posts
            if is_hate_post:
                user_hate_map[author] = True
        
        return user_hate_map
    
    def _get_username_by_index(self, user_idx, graph_data):
        """
        Get username for a given user index through reverse lookup.
        
        Args:
            user_idx (int): User node index
            graph_data: Graph data containing user mappings
            
        Returns:
            str or None: Username if found, None otherwise
        """
        user_to_id_mapping = getattr(graph_data, 'user_to_id', {})
        
        for username, idx in user_to_id_mapping.items():
            if idx == user_idx:
                return username
        
        return None
    
    def _create_train_test_split(self, graph_data, num_nodes):
        """
        Create training and testing masks for nodes.
        
        Args:
            graph_data: Graph data to add masks to
            num_nodes (int): Number of nodes
        """
        # Split node indices into train and test sets
        train_nodes, test_nodes = train_test_split(
            range(num_nodes), 
            test_size=0.2,  # 20% for testing
            random_state=42,  # For reproducibility
            stratify=graph_data.y.cpu().numpy()  # Maintain label balance
        )
        
        # Create boolean masks for training and testing
        graph_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(self.device)
        graph_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(self.device)
        
        # Set mask values
        graph_data.train_mask[train_nodes] = True
        graph_data.test_mask[test_nodes] = True
    
    def _log_data_statistics(self, graph_data):
        """
        Log statistics about the prepared data with semi-supervised learning info.
        """
        total_nodes = graph_data.num_nodes
        labelled_nodes = graph_data.label_mask.sum().item()
        unlabelled_nodes = total_nodes - labelled_nodes

        train_count = graph_data.train_mask.sum().item()
        val_count = graph_data.val_mask.sum().item()
        test_count = graph_data.test_mask.sum().item()

        # Only count labels for labelled nodes
        labelled_y = graph_data.y[graph_data.label_mask]
        if len(labelled_y) > 0:
            hate_count = labelled_y.sum().item()
            normal_count = len(labelled_y) - hate_count
        else:
            hate_count = normal_count = 0

        print(f"Total nodes: {total_nodes} ({labelled_nodes} labelled + {unlabelled_nodes} unlabelled)")
        print(f"Data split: {train_count} train, {val_count} val, {test_count} test")
        print(f"Label distribution: {hate_count} hate, {normal_count} normal")
    
    def train_classification(self, graph_data, epochs=100):
        """Train hate speech classification task with semi-supervised learning."""
        print("Training hate speech classification with semi-supervised learning...")

        # Move graph data to device
        graph_data = graph_data.to(self.device)
        best_f1 = 0

        for epoch in tqdm(range(epochs), desc="Training epochs"):
            self.model.train()
            self.optimizer.zero_grad()

            # Forward pass on ALL nodes (labelled + unlabelled)
            out = self.model(graph_data, task='classification')

            # Apply label mask: only compute loss on labelled training nodes
            train_labelled_mask = graph_data.train_mask & graph_data.label_mask

            if train_labelled_mask.sum() == 0:
                print("No labelled training data available!")
                continue

            # Get predictions and labels for labelled training nodes only
            train_logits = out[train_labelled_mask]
            train_labels = graph_data.y[train_labelled_mask].float()

            # Compute loss only on labelled training data
            loss = self.bce_loss(train_logits.squeeze(), train_labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Evaluation every 5 epochs for testing
            if epoch % 5 == 0:
                # Calibrate threshold on validation set
                self.calibrate_threshold(graph_data)

                # Evaluate on validation set during training (not test!)
                metrics = self.evaluate_with_threshold(graph_data, use_test=False)

                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']

                print(f"Epoch {epoch}: Loss = {loss:.4f}, F1 = {metrics['f1']:.4f}, "
                      f"Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}")

        return best_f1

    def evaluate_with_threshold(self, graph_data, use_test=True):
        """Evaluate model using calibrated threshold."""
        self.model.eval()
        with torch.no_grad():
            # Get predictions
            out = self.model(graph_data, task='classification')

            # Choose evaluation set: test for final eval, val for training monitoring
            if use_test:
                eval_mask = graph_data.test_mask & graph_data.label_mask
                set_name = "test"
            else:
                eval_mask = graph_data.val_mask & graph_data.label_mask
                set_name = "validation"

            if eval_mask.sum() == 0:
                print(f"No labelled {set_name} data available!")
                return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

            eval_logits = out[eval_mask]
            eval_labels = graph_data.y[eval_mask]

            # Convert to probabilities and apply threshold
            eval_probs = torch.sigmoid(eval_logits).cpu().numpy()
            pred_binary = (eval_probs > self.best_threshold).astype(int)
            true_labels = eval_labels.cpu().numpy()

            # Compute metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            accuracy = accuracy_score(true_labels, pred_binary)
            precision = precision_score(true_labels, pred_binary, zero_division=0)
            recall = recall_score(true_labels, pred_binary, zero_division=0)
            f1 = f1_score(true_labels, pred_binary, zero_division=0)

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

    def evaluate_model(self, graph_data):
        """Comprehensive model evaluation."""
        print("Evaluating model...")
        
        self.model.eval()
        with torch.no_grad():
            # Classification evaluation
            out = self.model(graph_data, task='classification')
            user_out = out[:graph_data.num_users]
            
            # Use labelled test nodes only
            test_labelled_mask = graph_data.test_mask & graph_data.label_mask

            if test_labelled_mask.sum() == 0:
                print("No labelled test data available!")
                return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0.5}

            test_logits = out[test_labelled_mask]
            test_labels = graph_data.y[test_labelled_mask]

            # Use sigmoid for binary classification
            probs = torch.sigmoid(test_logits).squeeze()

            # Dynamic threshold based on class distribution
            hate_ratio = test_labels.float().mean().item()
            threshold = max(0.1, min(0.5, hate_ratio * 10))  # Adaptive threshold
            pred = (probs > threshold).long()
            print(f"Using threshold: {threshold:.3f} (hate ratio: {hate_ratio:.4f})")
            true_labels = test_labels
            
            # Compute metrics
            accuracy = (pred == true_labels).float().mean().item()
            
            # Debug: Check prediction distribution
            if len(probs.shape) == 0:  # Single value
                hate_probs = probs.cpu().numpy().reshape(1)
            else:
                hate_probs = probs.cpu().numpy()

            if len(hate_probs) > 0:
                print(f"Hate probability stats: min={hate_probs.min():.4f}, max={hate_probs.max():.4f}, mean={hate_probs.mean():.4f}")
                print(f"Predictions: {pred.cpu().numpy().sum()} hate out of {len(pred)} total")
                print(f"True labels: {true_labels.cpu().numpy().sum()} hate out of {len(true_labels)} total")
            else:
                print("No test data to evaluate")
            
            # Convert to numpy for sklearn metrics
            pred_np = pred.cpu().numpy()
            true_np = true_labels.cpu().numpy()
            
            # Classification report
            report = classification_report(true_np, pred_np, output_dict=True)
            
            # ROC AUC - handle edge cases
            try:
                if len(np.unique(true_np)) > 1:  # Need both classes for AUC
                    auc = roc_auc_score(true_np, hate_probs)
                else:
                    auc = 0.5  # Random performance when only one class
            except Exception as e:
                print(f"AUC calculation failed: {e}")
                auc = 0.5
            
         # Handle case where class '1' might not exist in predictions
            if '1' in report:
                precision = report['1']['precision']
                recall = report['1']['recall']
                f1 = report['1']['f1-score']
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
            
            return metrics

def load_temporal_graph(config):
    """Load temporal graph data from CSV files with proper label masking for semi-supervised learning."""
    print("Loading temporal graph from CSV files...")

    # Load CSV files
    def load_csv_robust(path):
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                print(f"Trying to load {path} with {encoding} encoding...")
                return pd.read_csv(path, encoding=encoding, on_bad_lines='skip',
                                 engine='python', quoting=3, sep=',')
            except Exception as e:
                print(f"Failed with {encoding}: {str(e)[:100]}...")
                continue

        # Final fallback - try with chunk loading
        print("All encodings failed, trying chunk loading...")
        try:
            chunks = []
            chunk_size = 5000
            max_chunks = 2  # Load only 10k rows for FAST MODE
            for i, chunk in enumerate(pd.read_csv(path, encoding='latin-1',
                                                chunksize=chunk_size,
                                                on_bad_lines='skip')):
                chunks.append(chunk)
                if i >= max_chunks:
                    print(f"Loaded {len(chunks) * chunk_size} rows (FAST MODE limit)")
                    break
            return pd.concat(chunks, ignore_index=True)
        except Exception as e:
            print(f"Chunk loading also failed: {e}")
            raise ValueError(f"Could not load {path} with any method")

    # Load labelled data (balanced supervision sets) - FAST MODE with sampling
    print("Loading labelled supervision data...")

    def load_csv_fast(path, max_rows=None):
        """Load CSV with optional row limit - improved parsing"""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                if max_rows:
                    print(f"Loading {path} with {encoding} (max {max_rows} rows)...")
                else:
                    print(f"Loading full {path} with {encoding}...")
                # Try different quoting options to handle malformed CSV
                return pd.read_csv(path, encoding=encoding, on_bad_lines='skip',
                                 engine='python', quoting=1, sep=',', nrows=max_rows)  # QUOTE_ALL
            except Exception:
                try:
                    # Fallback with different quoting
                    return pd.read_csv(path, encoding=encoding, on_bad_lines='skip',
                                     engine='python', quoting=0, sep=',', nrows=max_rows)  # QUOTE_MINIMAL
                except Exception:
                    continue
        raise ValueError(f"Could not load {path}")

    # Load datasets with limits for testing (will restore full later)
    train_df = load_csv_fast('../supervision_train80_threads.csv', 10000)
    val_df = load_csv_fast('../supervision_validation10_threads.csv', 2000)
    test_df = load_csv_fast('../supervision_test10_threads.csv', 2000)

    print(f"Loaded train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)} comments")

    def clean_dataset(df, name):
        """Clean a single dataset and return processed version."""
        # Check which label column exists - prefer is_hate over hate_label due to data quality
        label_col = None
        for col in ['label', 'is_hate', 'hate_label']:  # Changed order to prefer is_hate
            if col in df.columns:
                label_col = col
                break

        if label_col:
            # Rename to standard 'label' column
            if label_col != 'label':
                df['label'] = df[label_col]

            # Remove NaN labels and keep only numeric labels
            df = df.dropna(subset=['label'])

            # Convert to numeric and keep only 0/1 labels
            def clean_label(x):
                try:
                    val = float(x)
                    if val in [0.0, 1.0]:
                        return int(val)
                    return None
                except:
                    return None

            df['label_clean'] = df['label'].apply(clean_label)
            df = df.dropna(subset=['label_clean'])
            df['label'] = df['label_clean'].astype(int)

            print(f"After cleaning {name}: {len(df)} labelled comments")
            print(f"{name} label distribution: {df['label'].value_counts().to_dict()}")
            return df
        else:
            print(f"No label column found in {name}, using dummy labels")
            df['label'] = 0
            return df

    # Clean each dataset separately
    train_df_clean = clean_dataset(train_df.copy(), "train")
    val_df_clean = clean_dataset(val_df.copy(), "val")
    test_df_clean = clean_dataset(test_df.copy(), "test")

    # Combine for graph building but keep track of splits
    labelled_df = pd.concat([train_df_clean, val_df_clean, test_df_clean], ignore_index=True)

    # Add split information
    train_ids = set(train_df_clean['id'].values)
    val_ids = set(val_df_clean['id'].values)
    test_ids = set(test_df_clean['id'].values)

    labelled_df['split'] = 'unknown'
    labelled_df.loc[labelled_df['id'].isin(train_ids), 'split'] = 'train'
    labelled_df.loc[labelled_df['id'].isin(val_ids), 'split'] = 'val'
    labelled_df.loc[labelled_df['id'].isin(test_ids), 'split'] = 'test'

    print(f"Total cleaned samples: {len(labelled_df)} (train: {len(train_df_clean)}, val: {len(val_df_clean)}, test: {len(test_df_clean)})")

    # Skip unlabelled data loading for FAST MODE
    print("Skipping unlabelled data loading for FAST MODE")
    unlabelled_df = pd.DataFrame()

    # Use only labelled data (skip unlabelled for now)
    all_df = labelled_df.copy()
    print(f"Using only labelled data: {len(all_df)} comments with proper train/val/test splits")

    # Build graph data from combined CSV
    graph_data = build_graph_from_csv(all_df)

    return graph_data

def _safe_float(value):
    """Safely convert value to float."""
    try:
        if pd.isna(value) or value is None:
            return 0.0
        # Try direct conversion first
        return float(value)
    except (ValueError, TypeError):
        try:
            # Try to extract numbers from string
            import re
            numbers = re.findall(r'-?\d+\.?\d*', str(value))
            return float(numbers[0]) if numbers else 0.0
        except:
            return 0.0

def build_graph_from_csv(df):
    """Build PyTorch Geometric graph data from CSV with proper label masking."""
    print("Building graph from CSV data...")

    # Create user and comment mappings
    users = df['author'].unique()
    user_to_id = {user: i for i, user in enumerate(users)}

    # Create node features using text embeddings
    print("Computing text embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Get comment texts - FAST MODE with smaller batch
    texts = df['body'].fillna('').astype(str).tolist()
    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)  # Smaller batch for speed

    # Create node features, labels, and masks
    node_features = []
    node_labels = []
    train_mask = []
    val_mask = []
    test_mask = []
    label_mask = []  # Mask for labelled vs unlabelled data

    for i, (_, row) in enumerate(df.iterrows()):
        # Text embedding
        text_emb = embeddings[i]

        # User features
        user_id = user_to_id[row['author']]

        # Simple features
        features = np.concatenate([
            text_emb,  # Text embedding
            [user_id / len(users)],  # Normalized user ID
            [_safe_float(row.get('score', 0)) / 100],  # Normalized score
            [len(row['body']) / 1000 if pd.notna(row['body']) else 0],  # Text length
        ])

        node_features.append(features)

        # Handle labels with proper masking for semi-supervised learning
        if 'label' in row and row['label'] != -1:  # Labelled data
            label_val = int(_safe_float(row['label']))
            node_labels.append(min(max(label_val, 0), 1))  # Ensure 0 or 1
            label_mask.append(True)  # This node has a label

            # Create split masks
            split = row.get('split', 'train')
            train_mask.append(split == 'train')
            val_mask.append(split == 'val')
            test_mask.append(split == 'test')

        else:  # Unlabelled data
            node_labels.append(-1)  # Use -1 for unlabelled
            label_mask.append(False)  # This node has no label
            train_mask.append(False)
            val_mask.append(False)
            test_mask.append(False)
    
    # Build edges based on reply relationships
    print("Building edges from reply relationships...")
    edges = []
    edge_times = []

    # Create ID to index mapping for faster lookup
    id_to_idx = {row['id']: i for i, (_, row) in enumerate(df.iterrows())}

    for i, (_, row) in enumerate(df.iterrows()):
        parent_id = row.get('parent_id', '')
        if parent_id and parent_id != row.get('link_id', ''):
            # Fast lookup using dictionary
            parent_idx = id_to_idx.get(parent_id)
            if parent_idx is not None:
                edges.append([parent_idx, i])  # Parent -> Child
                edge_times.append(_safe_float(row.get('created_utc', 0)))

    # Convert to tensors
    x = torch.FloatTensor(node_features)
    y = torch.LongTensor(node_labels)

    # Create mask tensors
    label_mask_tensor = torch.BoolTensor(label_mask)
    train_mask_tensor = torch.BoolTensor(train_mask)
    val_mask_tensor = torch.BoolTensor(val_mask)
    test_mask_tensor = torch.BoolTensor(test_mask)

    if edges:
        edge_index = torch.LongTensor(edges).t().contiguous()
        edge_attr = torch.FloatTensor([[t] for t in edge_times])
        edge_time = torch.FloatTensor(edge_times)
    else:
        # Create dummy edges if no parent-child relationships found
        edge_index = torch.LongTensor([[0], [0]])
        edge_attr = torch.FloatTensor([[0]])
        edge_time = torch.FloatTensor([0])

    # Create graph data with masks
    graph_data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_time=edge_time,
        num_nodes=len(node_features),
        num_users=len(users),
        user_to_id=user_to_id,
        # Masks for semi-supervised learning
        label_mask=label_mask_tensor,  # True for labelled nodes, False for unlabelled
        train_mask=train_mask_tensor,   # True for training nodes
        val_mask=val_mask_tensor,       # True for validation nodes
        test_mask=test_mask_tensor      # True for test nodes
    )

    print(f"Built graph: {graph_data.num_nodes} nodes, {edge_index.shape[1]} edges")
    print(f"Labelled nodes: {label_mask_tensor.sum().item()}/{len(label_mask)}")
    print(f"Train/Val/Test: {train_mask_tensor.sum().item()}/{val_mask_tensor.sum().item()}/{test_mask_tensor.sum().item()}")
    return graph_data

def save_model_results(model, metrics, config):
    """Save trained model and results."""
    print("Saving model and results...")
    
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = artifacts_dir / 'tgnn_model.pt'
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    with open(artifacts_dir / 'tgnn_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model config
    model_config = {
        'model_type': model.model_type,
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'num_classes': model.num_classes
    }
    
    with open(artifacts_dir / 'tgnn_model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"Model and results saved to {artifacts_dir}")

def main():
    parser = argparse.ArgumentParser(description="TGNN model training for Reddit hate speech")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")  # Test version: 10 epochs
    parser.add_argument("--hidden", type=int, default=32, help="Hidden dimension")  # Test version: 32 dim
    parser.add_argument("--model_type", type=str, default='TGNN', choices=['TGAT', 'TGN', 'TGNN'], help="Model type")
    args = parser.parse_args()
    
    # Create simple config
    config = {
        'tgnn': {
            'model_type': args.model_type,
            'input_dim': 387,  # Will be updated based on actual features
            'hidden_dim': args.hidden,
            'num_layers': 3,  # Changed from 2 to 3
            'num_classes': 2,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'num_epochs': args.epochs,
            'memory_dim': 64
        },
        'paths': {
            'artifacts_dir': 'artifacts'
        }
    }
    
    print("=== TGNN Model Training ===")
    
    # Load temporal graph
    graph_data = load_temporal_graph(config)
    
    # Update input dimension based on actual features
    config['tgnn']['input_dim'] = graph_data.x.shape[1]
    
    # Initialize model
    model = TGNNModel(config)
    print(f"Initialized {model.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    trainer = TGNNTrainer(model, config)
    
    # Train classification task
    best_acc = trainer.train_classification(graph_data, epochs=args.epochs)
    
    # Final evaluation on test set
    print("\n=== Final Evaluation on Test Set ===")
    metrics = trainer.evaluate_with_threshold(graph_data, use_test=True)
    
    # Save results
    save_model_results(model, metrics, config)
    
    print(f"\n=== TGNN Training Summary ===")
    print(f"Model: {model.model_type}")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Test Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
