#!/usr/bin/env python3
"""
Step 2 - Feature Engineering Script for Reddit Hate Speech Analysis.
Implements node generation, graph generation, and feature extraction.
"""

import argparse
import json
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

warnings.filterwarnings('ignore')

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class Step2FeatureExtractor:
    """Step 2 Feature Engineering for Reddit Hate Speech Analysis."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading sentence transformer model: {model_name}")
        print(f"Using device: {self.device}")
        print(f"Batch size: {batch_size}")

        # Load sentence transformer model
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)

        print(f"Sentence transformer model loaded successfully!")

    def extract_comment_embeddings(self, texts):
        """Extract comment vector embeddings using sentence-transformers."""
        print(f"Extracting embeddings for {len(texts)} comments...")

        # Process in batches
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=True)
            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def generate_user_features(self, comments_df):
        """Generate User node features."""
        print("Generating User node features...")

        user_features = {}

        for user in tqdm(comments_df['author'].unique(), desc="Processing users"):
            if pd.isna(user) or user == '[deleted]':
                continue

            user_comments = comments_df[comments_df['author'] == user]

            # Features: unique subreddits posted to, average posting time interval, hate speech ratio
            unique_subreddits = user_comments['subreddit'].nunique()

            # Calculate average posting time interval
            if len(user_comments) > 1:
                time_intervals = user_comments['created_utc'].sort_values().diff().dropna()
                avg_time_interval = time_intervals.mean() if len(time_intervals) > 0 else 0
            else:
                avg_time_interval = 0

            # Calculate hate speech ratio
            total_comments = len(user_comments)
            hate_comments = len(user_comments[user_comments['cardiffnlp_label'] == 1])
            hate_speech_ratio = hate_comments / total_comments if total_comments > 0 else 0

            user_features[user] = {
                'unique_subreddits': unique_subreddits,
                'avg_posting_interval': avg_time_interval,
                'hate_speech_ratio': hate_speech_ratio,
                'total_comments': total_comments,
                'hate_comments': hate_comments
            }

        return user_features

    def generate_subreddit_features(self, comments_df):
        """Generate Subreddit node features."""
        print("Generating Subreddit node features...")

        subreddit_features = {}

        for subreddit in tqdm(comments_df['subreddit'].unique(), desc="Processing subreddits"):
            if pd.isna(subreddit):
                continue

            subreddit_comments = comments_df[comments_df['subreddit'] == subreddit]

            # Features: average posting time interval, hate speech ratio
            # Calculate average posting time interval
            if len(subreddit_comments) > 1:
                time_intervals = subreddit_comments['created_utc'].sort_values().diff().dropna()
                avg_time_interval = time_intervals.mean() if len(time_intervals) > 0 else 0
            else:
                avg_time_interval = 0

            # Calculate hate speech ratio
            total_comments = len(subreddit_comments)
            hate_comments = len(subreddit_comments[subreddit_comments['cardiffnlp_label'] == 1])
            hate_speech_ratio = hate_comments / total_comments if total_comments > 0 else 0

            subreddit_features[subreddit] = {
                'avg_posting_interval': avg_time_interval,
                'hate_speech_ratio': hate_speech_ratio,
                'total_comments': total_comments,
                'hate_comments': hate_comments
            }

        return subreddit_features

    def generate_comment_features(self, comments_df):
        """Generate Comment node features."""
        print("Generating Comment node features...")

        # Extract embeddings for all comments
        comment_embeddings = self.extract_comment_embeddings(comments_df['text_content'].tolist())

        # Calculate token length for each comment
        token_lengths = comments_df['text_content'].str.split().str.len().fillna(0)

        comment_features = {}
        for idx, (_, comment) in enumerate(comments_df.iterrows()):
            comment_features[comment['id']] = {
                'embedding': comment_embeddings[idx],
                'token_length': token_lengths.iloc[idx],
                'score': comment.get('score', 0),
                'cardiffnlp_label': comment.get('cardiffnlp_label', 0)
            }

        return comment_features

    def generate_graph(self, comments_df, user_features, subreddit_features, comment_features):
        """Generate graph with User-Comment, Comment-Comment, User-Subreddit edges."""
        print("Generating graph structure...")

        G = nx.DiGraph()

        # Add nodes with features
        print("Adding User nodes...")
        for user, features in user_features.items():
            G.add_node(user, node_type='user', **features)

        print("Adding Subreddit nodes...")
        for subreddit, features in subreddit_features.items():
            G.add_node(subreddit, node_type='subreddit', **features)

        print("Adding Comment nodes...")
        for comment_id, features in comment_features.items():
            G.add_node(comment_id, node_type='comment', **features)

        # Add edges
        print("Adding User-Comment edges (posted)...")
        for _, comment in tqdm(comments_df.iterrows(), total=len(comments_df), desc="User-Comment edges"):
            if not pd.isna(comment['author']) and comment['author'] != '[deleted]':
                G.add_edge(comment['author'], comment['id'], edge_type='posted')

        print("Adding Comment-Comment edges (in_response_to)...")
        for _, comment in tqdm(comments_df.iterrows(), total=len(comments_df), desc="Comment-Comment edges"):
            if not pd.isna(comment['parent_id']) and comment['parent_id'] in comment_features:
                G.add_edge(comment['parent_id'], comment['id'], edge_type='in_response_to')

        print("Adding User-Subreddit edges (participated_in)...")
        for _, comment in tqdm(comments_df.iterrows(), total=len(comments_df), desc="User-Subreddit edges"):
            if not pd.isna(comment['author']) and comment['author'] != '[deleted]' and not pd.isna(comment['subreddit']):
                G.add_edge(comment['author'], comment['subreddit'], edge_type='participated_in')

        print(f"Graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def generate_numeric_embeddings(self, comments_df):
        """Generate numeric embeddings for Author and Subreddit."""
        print("Generating numeric embeddings for Author and Subreddit...")

        # Create mappings for authors and subreddits
        unique_authors = comments_df['author'].dropna().unique()
        unique_subreddits = comments_df['subreddit'].dropna().unique()

        # Remove '[deleted]' from authors
        unique_authors = [a for a in unique_authors if a != '[deleted]']

        # Create numeric mappings
        author_to_id = {author: idx for idx, author in enumerate(unique_authors)}
        subreddit_to_id = {subreddit: idx for idx, subreddit in enumerate(unique_subreddits)}

        # Generate embeddings for comments
        comment_embeddings = {}
        for _, comment in comments_df.iterrows():
            author_id = author_to_id.get(comment['author'], -1) if not pd.isna(comment['author']) else -1
            subreddit_id = subreddit_to_id.get(comment['subreddit'], -1) if not pd.isna(comment['subreddit']) else -1

            comment_embeddings[comment['id']] = {
                'author_id': author_id,
                'subreddit_id': subreddit_id,
                'n_votes': comment.get('score', 0)
            }

        return {
            'author_to_id': author_to_id,
            'subreddit_to_id': subreddit_to_id,
            'comment_embeddings': comment_embeddings
        }

def save_features(user_features, subreddit_features, comment_features, graph, numeric_embeddings, config):
    """Save all generated features."""
    print("Saving Step 2 features...")

    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)

    # Save user features
    user_features_path = artifacts_dir / 'step2_user_features.json'
    with open(user_features_path, 'w', encoding='utf-8') as f:
        json.dump(user_features, f, indent=2, default=str)
    print(f"User features saved to {user_features_path}")

    # Save subreddit features
    subreddit_features_path = artifacts_dir / 'step2_subreddit_features.json'
    with open(subreddit_features_path, 'w', encoding='utf-8') as f:
        json.dump(subreddit_features, f, indent=2, default=str)
    print(f"Subreddit features saved to {subreddit_features_path}")

    # Save comment features (without embeddings for JSON)
    comment_features_json = {}
    for comment_id, features in comment_features.items():
        comment_features_json[comment_id] = {
            'token_length': features['token_length'],
            'score': features['score'],
            'cardiffnlp_label': features['cardiffnlp_label']
        }

    comment_features_path = artifacts_dir / 'step2_comment_features.json'
    with open(comment_features_path, 'w', encoding='utf-8') as f:
        json.dump(comment_features_json, f, indent=2, default=str)
    print(f"Comment features saved to {comment_features_path}")

    # Save comment embeddings separately
    comment_embeddings = {comment_id: features['embedding'].tolist()
                         for comment_id, features in comment_features.items()}
    comment_embeddings_path = artifacts_dir / 'step2_comment_embeddings.json'
    with open(comment_embeddings_path, 'w', encoding='utf-8') as f:
        json.dump(comment_embeddings, f, indent=2)
    print(f"Comment embeddings saved to {comment_embeddings_path}")

    # Save graph
    graph_path = artifacts_dir / 'step2_graph.pkl'
    import pickle
    with open(graph_path, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Graph saved to {graph_path}")

    # Save numeric embeddings
    numeric_embeddings_path = artifacts_dir / 'step2_numeric_embeddings.json'
    with open(numeric_embeddings_path, 'w', encoding='utf-8') as f:
        json.dump(numeric_embeddings, f, indent=2, default=str)
    print(f"Numeric embeddings saved to {numeric_embeddings_path}")

def main():
    print("=== Step 2: Feature Engineering ===")

    parser = argparse.ArgumentParser(description="Step 2 - Feature Engineering for Reddit Hate Speech Analysis")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load Step 1 data
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    comments_df = pd.read_parquet(artifacts_dir / 'step1_labeled_comments.parquet')

    print(f"Loaded {len(comments_df)} comments from Step 1")

    # Initialize feature extractor
    extractor = Step2FeatureExtractor(
        model_name=config.get('bert', {}).get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
        batch_size=config.get('bert', {}).get('batch_size', 32)
    )

    # Generate node features
    print("\n=== Node Generation ===")
    user_features = extractor.generate_user_features(comments_df)
    subreddit_features = extractor.generate_subreddit_features(comments_df)
    comment_features = extractor.generate_comment_features(comments_df)

    # Generate graph
    print("\n=== Graph Generation ===")
    graph = extractor.generate_graph(comments_df, user_features, subreddit_features, comment_features)

    # Generate numeric embeddings
    print("\n=== Numeric Embeddings ===")
    numeric_embeddings = extractor.generate_numeric_embeddings(comments_df)

    # Save all features
    print("\n=== Saving Features ===")
    save_features(user_features, subreddit_features, comment_features, graph, numeric_embeddings, config)

    print("\n=== Step 2 Summary ===")
    print(f"User nodes: {len(user_features)}")
    print(f"Subreddit nodes: {len(subreddit_features)}")
    print(f"Comment nodes: {len(comment_features)}")
    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    print(f"Unique authors: {len(numeric_embeddings['author_to_id'])}")
    print(f"Unique subreddits: {len(numeric_embeddings['subreddit_to_id'])}")

    print("\nStep 2 - Feature Engineering completed successfully!")
    print("Ready for Step 3 - Model Training")

if __name__ == "__main__":
    main()
