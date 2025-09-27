#!/usr/bin/env python3
"""
Temporal Graph Construction Script for Reddit Hate Speech Analysis.
Builds temporal graphs with comment-only nodes and user/subreddit as features.
"""

import pandas as pd
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data, TemporalData
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class TemporalGraphBuilder:
    """Builder for temporal graphs with comment-only nodes."""
    
    def __init__(self, config):
        self.config = config
        self.artifacts_dir = Path(config['paths']['artifacts_dir'])
        self.df = None
        self.embeddings = None
        self.user_features = None
        
        # Node mappings
        self.comment_to_id = {}
        self.submission_to_id = {}
        self.node_types = {}  # 0=comment, 1=submission placeholder
        
        # Graph data
        self.node_features = None
        self.edge_index = None
        self.edge_attr = None
        self.edge_timestamps = None
        
    def load_data(self):
        """Load prepared data and embeddings."""
        print("Loading prepared data...")
        
        # 优先加载thread paired数据集（更大更完整）
        thread_path = self.artifacts_dir / 'thread_paired_dataset.parquet'
        balanced_path = self.artifacts_dir / 'balanced_dataset.parquet'
        
        if thread_path.exists():
            self.df = pd.read_parquet(thread_path)
            print(f"Loaded thread paired dataset: {len(self.df)} samples")
        elif balanced_path.exists():
            self.df = pd.read_parquet(balanced_path)
            print(f"Loaded balanced dataset: {len(self.df)} samples")
        else:
            raise FileNotFoundError("No prepared dataset found!")
        
        # Load BERT embeddings
        embeddings_path = self.artifacts_dir / 'bert_embeddings.npy'
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
            print(f"Loaded embeddings: {self.embeddings.shape}")
        else:
            raise FileNotFoundError("BERT embeddings not found! Run BERT feature extraction first.")
        
        # Load user features
        user_features_path = self.artifacts_dir / 'user_features.json'
        if user_features_path.exists():
            with open(user_features_path, 'r') as f:
                self.user_features = json.load(f)
            print(f"Loaded user features for {len(self.user_features)} users")
        else:
            print("Warning: User features not found, using defaults")
            self.user_features = {}
    
    def create_node_mappings(self):
        """Create node ID mappings for comment-only structure."""
        print("Creating comment-only node mappings...")
        
        # Create comment nodes
        for idx, comment_id in enumerate(self.df['id'].unique()):
            self.comment_to_id[comment_id] = idx
        
        # Create submission placeholder nodes for root comments
        submission_count = 0
        for link_id in self.df['link_id'].unique():
            if link_id not in self.comment_to_id:  # This is a submission, not a comment
                self.submission_to_id[link_id] = len(self.comment_to_id) + submission_count
                submission_count += 1
        
        total_nodes = len(self.comment_to_id) + len(self.submission_to_id)
        print(f"Created {len(self.comment_to_id)} comment nodes, {len(self.submission_to_id)} submission placeholder nodes")
        print(f"Total nodes: {total_nodes}")
        
        return total_nodes
    
    def create_node_features(self, total_nodes):
        """Create node feature matrix for comment-only structure with user/subreddit as features."""
        print("Creating comment-only node features...")
        
        # Initialize feature matrix with node type encoding
        # BERT embeddings + user features + subreddit features + node type one-hot
        feature_dim = self.embeddings.shape[1] + 8 + 3 + 2  # BERT + user + subreddit + node type
        node_features = np.zeros((total_nodes, feature_dim))
        self.node_types = np.zeros(total_nodes, dtype=int)
        
        # Process comment nodes (type 0)
        for comment, comment_id in tqdm(self.comment_to_id.items(), desc="Processing comment nodes"):
            # Get comment data
            comment_data = self.df[self.df['id'] == comment]
            
            if len(comment_data) > 0:
                comment_idx = comment_data.index[0]
                if comment_idx < len(self.embeddings):
                    # BERT embedding
                    node_features[comment_id, :self.embeddings.shape[1]] = self.embeddings[comment_idx]
                
                # Add user features (as node features)
                comment_row = comment_data.iloc[0]
                author = comment_row.get('author', '')
                base_idx = self.embeddings.shape[1]
                
                if author in self.user_features:
                    uf = self.user_features[author]
                    node_features[comment_id, base_idx:base_idx+8] = [
                        uf.get('total_posts', 0),
                        uf.get('hate_ratio', 0),
                        uf.get('offensive_ratio', 0),
                        uf.get('subreddit_diversity', 0),
                        uf.get('avg_posting_interval_hours', 0),
                        uf.get('avg_text_length', 0),
                        comment_row.get('davidson_label', 0) == 2,  # is_hate
                        len(comment_row.get('text_content', ''))  # text length
                    ]
                else:
                    # Default user features if not found
                    node_features[comment_id, base_idx:base_idx+8] = [
                        1.0,  # total_posts
                        0.0,  # hate_ratio
                        0.0,  # offensive_ratio
                        1.0,  # subreddit_diversity
                        0.0,  # avg_posting_interval_hours
                        len(comment_row.get('text_content', '')),  # avg_text_length
                        comment_row.get('davidson_label', 0) == 2,  # is_hate
                        len(comment_row.get('text_content', ''))  # text length
                    ]
                
                # Add subreddit features (as node features)
                subreddit = comment_row.get('subreddit', '')
                subreddit_base_idx = base_idx + 8
                subreddit_data = self.df[self.df['subreddit'] == subreddit]
                if len(subreddit_data) > 0:
                    node_features[comment_id, subreddit_base_idx:subreddit_base_idx+3] = [
                        len(subreddit_data),  # total posts in subreddit
                        subreddit_data['davidson_label'].apply(lambda x: x == 2).mean(),  # hate ratio
                        subreddit_data['davidson_label'].apply(lambda x: x == 1).mean()   # offensive ratio
                    ]
                else:
                    node_features[comment_id, subreddit_base_idx:subreddit_base_idx+3] = [1.0, 0.0, 0.0]
                
                # Set node type (comment = 0)
                self.node_types[comment_id] = 0
                node_features[comment_id, -2:] = [1.0, 0.0]  # Comment type one-hot
        
        # Process submission placeholder nodes (type 1)
        submission_offset = len(self.comment_to_id)
        for submission, submission_id in tqdm(self.submission_to_id.items(), desc="Processing submission placeholder nodes"):
            # Get submission data (from comments with this link_id)
            submission_data = self.df[self.df['link_id'] == submission]
            
            if len(submission_data) > 0:
                # Use average embedding for submission representation
                submission_indices = submission_data.index
                valid_indices = [idx for idx in submission_indices if idx < len(self.embeddings)]
                if valid_indices:
                    avg_embedding = np.mean(self.embeddings[valid_indices], axis=0)
                    node_features[submission_id, :self.embeddings.shape[1]] = avg_embedding
                
                # Add aggregated features
                base_idx = self.embeddings.shape[1]
                node_features[submission_id, base_idx:base_idx+8] = [
                    len(submission_data),  # total comments
                    submission_data['davidson_label'].apply(lambda x: x == 2).mean(),  # hate ratio
                    submission_data['davidson_label'].apply(lambda x: x == 1).mean(),  # offensive ratio
                    submission_data['subreddit'].nunique(),  # subreddit diversity
                    0.0,  # posting interval (N/A for submissions)
                    submission_data['text_content'].str.len().mean(),  # avg text length
                    submission_data['davidson_label'].apply(lambda x: x == 2).any(),  # has_hate
                    submission_data['text_content'].str.len().mean()  # avg text length
                ]
                
                # Add subreddit features
                subreddit_base_idx = base_idx + 8
                if len(submission_data) > 0:
                    subreddit = submission_data.iloc[0]['subreddit']
                    subreddit_data = self.df[self.df['subreddit'] == subreddit]
                    if len(subreddit_data) > 0:
                        node_features[submission_id, subreddit_base_idx:subreddit_base_idx+3] = [
                            len(subreddit_data),  # total posts in subreddit
                            subreddit_data['davidson_label'].apply(lambda x: x == 2).mean(),  # hate ratio
                            subreddit_data['davidson_label'].apply(lambda x: x == 1).mean()   # offensive ratio
                        ]
                    else:
                        node_features[submission_id, subreddit_base_idx:subreddit_base_idx+3] = [1.0, 0.0, 0.0]
                else:
                    node_features[submission_id, subreddit_base_idx:subreddit_base_idx+3] = [1.0, 0.0, 0.0]
                
                # Set node type (submission = 1)
                self.node_types[submission_id] = 1
                node_features[submission_id, -2:] = [0.0, 1.0]  # Submission type one-hot
        
        self.node_features = torch.FloatTensor(node_features)
        print(f"Created node features: {self.node_features.shape}")
        print(f"Node type distribution: {np.bincount(self.node_types)}")
    
    def create_temporal_edges(self):
        """Create temporal edges with timestamps."""
        print("Creating temporal edges...")
        
        edges = []
        edge_attrs = []
        edge_timestamps = []
        
        # Create Comment-Comment edges (reply relationships)
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Creating C-C edges"):
            comment_id = row['id']
            parent_id = row.get('parent_id', '')
            
            if parent_id and parent_id in self.comment_to_id:
                # Comment replies to another comment
                parent_comment_id = self.comment_to_id[parent_id]
                if comment_id in self.comment_to_id:
                    child_comment_id = self.comment_to_id[comment_id]
                    
                    # Edge from parent to child (reply relationship)
                    edges.append([parent_comment_id, child_comment_id])
                    
                    # Edge attributes: [edge_type, hate_label, subreddit_id, score, text_length]
                    edge_attrs.append([
                        0,  # edge_type: 0 = C-C reply
                        row.get('davidson_label', 0) == 2,  # is_hate
                        hash(row.get('subreddit', '')) % 1000,  # subreddit_id (hashed)
                        row.get('score', 0),  # score
                        len(row.get('text_content', ''))  # text_length
                    ])
                    
                    # Edge timestamp: use child comment's timestamp
                    edge_timestamps.append(row.get('created_utc', 0))
        
        # Create Comment-Submission edges (root comments to their submissions)
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Creating C-S edges"):
            comment_id = row['id']
            link_id = row.get('link_id', '')
            parent_id = row.get('parent_id', '')
            
            # Check if this is a root comment (no parent_id or parent_id is the submission)
            if link_id in self.submission_to_id and (not parent_id or parent_id == link_id):
                if comment_id in self.comment_to_id:
                    comment_node_id = self.comment_to_id[comment_id]
                    submission_node_id = self.submission_to_id[link_id]
                    
                    # Edge from submission to comment (root comment relationship)
                    edges.append([submission_node_id, comment_node_id])
                    
                    # Edge attributes
                    edge_attrs.append([
                        1,  # edge_type: 1 = C-S root
                        row.get('davidson_label', 0) == 2,  # is_hate
                        hash(row.get('subreddit', '')) % 1000,  # subreddit_id
                        row.get('score', 0),  # score
                        len(row.get('text_content', ''))  # text_length
                    ])
                    
                    # Edge timestamp: use comment's timestamp
                    edge_timestamps.append(row.get('created_utc', 0))
        
        if edges:
            self.edge_index = torch.LongTensor(edges).t().contiguous()
            self.edge_attr = torch.FloatTensor(edge_attrs)
            self.edge_timestamps = torch.LongTensor(edge_timestamps)
            
            print(f"Created {len(edges)} temporal edges")
            print(f"Edge types: C-C replies: {sum(1 for attr in edge_attrs if attr[0] == 0)}, C-S roots: {sum(1 for attr in edge_attrs if attr[0] == 1)}")
        else:
            print("Warning: No edges created!")
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            self.edge_attr = torch.empty((0, 5), dtype=torch.float)
            self.edge_timestamps = torch.empty(0, dtype=torch.long)
    
    def build_temporal_graph(self):
        """Build the complete temporal graph."""
        print("Building temporal graph...")
        
        # Create temporal graph data
        self.temporal_data = TemporalData(
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            edge_time=self.edge_timestamps,
            x=self.node_features
        )
        
        print(f"Temporal graph created:")
        print(f"  Nodes: {self.temporal_data.x.shape[0]}")
        print(f"  Edges: {self.edge_index.shape[1]}")
        print(f"  Node features: {self.temporal_data.x.shape[1]}")
        print(f"  Edge features: {self.edge_attr.shape[1]}")
        print(f"  Time range: {self.edge_timestamps.min().item()} - {self.edge_timestamps.max().item()}")
    
    def create_networkx_graph(self):
        """Create NetworkX graph for visualization and analysis."""
        print("Creating NetworkX graph...")
        
        G = nx.DiGraph()
        
        # Add comment nodes
        for comment, comment_id in self.comment_to_id.items():
            G.add_node(comment_id, 
                      type='comment',
                      id=comment,
                      node_type=0)
        
        # Add submission placeholder nodes
        for submission, submission_id in self.submission_to_id.items():
            G.add_node(submission_id,
                      type='submission',
                      id=submission,
                      node_type=1)
        
        # Add edges
        if self.edge_index is not None and self.edge_index.shape[1] > 0:
            for i in range(self.edge_index.shape[1]):
                src, dst = self.edge_index[:, i].tolist()
                edge_type = self.edge_attr[i, 0].item()
                timestamp = self.edge_timestamps[i].item()
                
                G.add_edge(src, dst,
                          edge_type=edge_type,
                          timestamp=timestamp)
        
        self.networkx_graph = G
        print(f"NetworkX graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def save_temporal_graph(self):
        """Save the temporal graph and related data."""
        print("Saving temporal graph...")
        
        # Save PyTorch Geometric temporal data
        torch.save(self.temporal_data, self.artifacts_dir / 'temporal_graph.pt')
        
        # Save NetworkX graph
        with open(self.artifacts_dir / 'temporal_graph_nx.pkl', 'wb') as f:
            pickle.dump(self.networkx_graph, f)
        
        # Save graph statistics
        graph_stats = {
            'num_nodes': self.temporal_data.x.shape[0],
            'num_edges': self.edge_index.shape[1] if self.edge_index is not None else 0,
            'num_comments': len(self.comment_to_id),
            'num_submissions': len(self.submission_to_id),
            'num_subreddits': self.df['subreddit'].nunique(),
            'node_feature_dim': self.temporal_data.x.shape[1],
            'edge_feature_dim': self.edge_attr.shape[1] if self.edge_attr is not None else 0,
            'time_range': {
                'min': self.edge_timestamps.min().item() if len(self.edge_timestamps) > 0 else 0,
                'max': self.edge_timestamps.max().item() if len(self.edge_timestamps) > 0 else 0
            },
            'node_type_distribution': np.bincount(self.node_types).tolist(),
            'edge_type_distribution': np.bincount(self.edge_attr[:, 0].int()).tolist() if self.edge_attr is not None else []
        }
        
        with open(self.artifacts_dir / 'temporal_graph_stats.json', 'w') as f:
            json.dump(graph_stats, f, indent=2)
        
        print("Temporal graph saved successfully!")
        print(f"Graph statistics: {graph_stats}")

def main():
    parser = argparse.ArgumentParser(description="Temporal graph construction for Reddit data")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("=== Temporal Graph Construction ===")
    
    # Initialize builder
    builder = TemporalGraphBuilder(config)
    
    # Load data
    builder.load_data()
    
    # Create node mappings
    total_nodes = builder.create_node_mappings()
    
    # Create node features
    builder.create_node_features(total_nodes)
    
    # Create temporal edges
    builder.create_temporal_edges()
    
    # Build temporal graph
    builder.build_temporal_graph()
    
    # Create NetworkX graph
    builder.create_networkx_graph()
    
    # Save everything
    builder.save_temporal_graph()
    
    print("Temporal graph construction completed!")

if __name__ == "__main__":
    main()
