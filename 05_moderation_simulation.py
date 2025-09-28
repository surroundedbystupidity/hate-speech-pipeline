#!/usr/bin/env python3
"""
Moderation Strategy Simulation Script for Reddit Hate Speech Analysis.
Simulates different moderation strategies and evaluates their effectiveness.
"""

import argparse
import copy
import json
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

warnings.filterwarnings('ignore')

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class ModerationSimulator:
    """Simulator for different moderation strategies."""

    def __init__(self, config):
        self.config = config
        self.moderation_config = config.get('moderation', {})
        self.hate_threshold = self.moderation_config.get('hate_threshold', 0.5)
        self.subreddit_hate_threshold = self.moderation_config.get('subreddit_hate_threshold', 0.3)

    def load_data(self):
        """Load all necessary data for simulation."""
        print("Loading data for moderation simulation...")

        artifacts_dir = Path(self.config['paths']['artifacts_dir'])

        # Load temporal graph
        graph_path = artifacts_dir / 'temporal_graph.pt'
        self.graph_data = torch.load(graph_path, weights_only=False)

        # Load NetworkX graph
        nx_path = artifacts_dir / 'temporal_graph_nx.pkl'
        with open(nx_path, 'rb') as f:
            self.original_graph = pickle.load(f)

        # Load enhanced dataset
        bert_path = artifacts_dir / 'bert_enhanced_dataset.parquet'
        if bert_path.exists():
            self.df = pd.read_parquet(bert_path)
        else:
            self.df = pd.read_parquet(artifacts_dir / 'balanced_dataset.parquet')

        # Load subreddit analysis
        subreddit_path = artifacts_dir / 'subreddit_analysis.json'
        with open(subreddit_path, 'r') as f:
            self.subreddit_analysis = json.load(f)

        print(f"Loaded graph: {self.original_graph.number_of_nodes()} nodes, {self.original_graph.number_of_edges()} edges")
        print(f"Loaded dataset: {len(self.df)} samples")

    def identify_hate_content(self):
        """Identify hate speech content and users."""
        print("Identifying hate speech content...")

        # Identify hate posts/comments
        hate_content = self.df[self.df['davidson_label'] == 2].copy()

        # Identify hate users (users with high hate ratio)
        hate_users = []
        for user, group in self.df.groupby('author'):
            if pd.isna(user):
                continue
            total_posts = len(group)
            hate_posts = len(group[group['davidson_label'] == 2])
            hate_ratio = hate_posts / total_posts if total_posts > 0 else 0

            if hate_ratio > self.hate_threshold and total_posts >= 3:
                hate_users.append({
                    'user': user,
                    'hate_ratio': hate_ratio,
                    'total_posts': total_posts,
                    'hate_posts': hate_posts
                })

        # Identify hate subreddits
        hate_subreddits = []
        for subreddit_info in self.subreddit_analysis['top_hate_subreddits']:
            if subreddit_info['hate_ratio'] > self.subreddit_hate_threshold:
                hate_subreddits.append(subreddit_info['subreddit'])

        self.hate_content = hate_content
        self.hate_users = hate_users
        self.hate_subreddits = hate_subreddits

        print(f"Identified {len(hate_content)} hate posts/comments")
        print(f"Identified {len(hate_users)} hate users")
        print(f"Identified {len(hate_subreddits)} hate subreddits")

        return {
            'hate_content_count': len(hate_content),
            'hate_users_count': len(hate_users),
            'hate_subreddits_count': len(hate_subreddits)
        }

    def simulate_content_removal(self, removal_percentage=0.2):
        """Simulate removing hate speech content."""
        print(f"Simulating content removal ({removal_percentage*100}% of hate content)...")

        # Create modified graph
        modified_graph = self.original_graph.copy()
        modified_df = self.df.copy()

        # Select content to remove
        hate_content_ids = self.hate_content['id'].tolist()
        num_to_remove = int(len(hate_content_ids) * removal_percentage)
        content_to_remove = np.random.choice(hate_content_ids, num_to_remove, replace=False)

        # Remove from dataframe
        modified_df = modified_df[~modified_df['id'].isin(content_to_remove)]

        # Remove corresponding nodes from graph (if they exist)
        post_to_id = getattr(self.graph_data, 'post_to_id', {})
        nodes_to_remove = []
        for content_id in content_to_remove:
            if content_id in post_to_id:
                node_id = post_to_id[content_id]
                if modified_graph.has_node(node_id):
                    nodes_to_remove.append(node_id)

        modified_graph.remove_nodes_from(nodes_to_remove)

        # Calculate impact
        original_hate_ratio = len(self.hate_content) / len(self.df)
        remaining_hate = modified_df[modified_df['davidson_label'] == 2]
        new_hate_ratio = len(remaining_hate) / len(modified_df) if len(modified_df) > 0 else 0

        impact = {
            'strategy': 'content_removal',
            'removal_percentage': removal_percentage,
            'content_removed': num_to_remove,
            'nodes_removed': len(nodes_to_remove),
            'original_hate_ratio': original_hate_ratio,
            'new_hate_ratio': new_hate_ratio,
            'hate_reduction': (original_hate_ratio - new_hate_ratio) / original_hate_ratio if original_hate_ratio > 0 else 0,
            'graph_connectivity': self.calculate_connectivity(modified_graph),
            'remaining_samples': len(modified_df)
        }

        return impact, modified_graph, modified_df

    def simulate_user_banning(self, ban_percentage=0.1):
        """Simulate banning hate speech users."""
        print(f"Simulating user banning ({ban_percentage*100}% of hate users)...")

        # Create modified graph and data
        modified_graph = self.original_graph.copy()
        modified_df = self.df.copy()

        # Select users to ban
        num_to_ban = int(len(self.hate_users) * ban_percentage)
        users_to_ban = np.random.choice(
            [user['user'] for user in self.hate_users],
            num_to_ban,
            replace=False
        )

        # Remove users' content from dataframe
        modified_df = modified_df[~modified_df['author'].isin(users_to_ban)]

        # Remove user nodes from graph
        user_to_id = getattr(self.graph_data, 'user_to_id', {})
        nodes_to_remove = []
        for user in users_to_ban:
            if user in user_to_id:
                node_id = user_to_id[user]
                if modified_graph.has_node(node_id):
                    nodes_to_remove.append(node_id)

        modified_graph.remove_nodes_from(nodes_to_remove)

        # Calculate impact
        original_hate_ratio = len(self.hate_content) / len(self.df)
        remaining_hate = modified_df[modified_df['davidson_label'] == 2]
        new_hate_ratio = len(remaining_hate) / len(modified_df) if len(modified_df) > 0 else 0

        impact = {
            'strategy': 'user_banning',
            'ban_percentage': ban_percentage,
            'users_banned': num_to_ban,
            'nodes_removed': len(nodes_to_remove),
            'original_hate_ratio': original_hate_ratio,
            'new_hate_ratio': new_hate_ratio,
            'hate_reduction': (original_hate_ratio - new_hate_ratio) / original_hate_ratio if original_hate_ratio > 0 else 0,
            'graph_connectivity': self.calculate_connectivity(modified_graph),
            'remaining_samples': len(modified_df)
        }

        return impact, modified_graph, modified_df

    def simulate_subreddit_banning(self, ban_percentage=0.1):
        """Simulate banning hate speech subreddits."""
        print(f"Simulating subreddit banning ({ban_percentage*100}% of hate subreddits)...")

        # Create modified graph and data
        modified_graph = self.original_graph.copy()
        modified_df = self.df.copy()

        # Select subreddits to ban
        if len(self.hate_subreddits) > 0:
            num_to_ban = max(1, int(len(self.hate_subreddits) * ban_percentage))
            subreddits_to_ban = np.random.choice(self.hate_subreddits, num_to_ban, replace=False)
        else:
            subreddits_to_ban = []
            num_to_ban = 0

        # Remove subreddit content from dataframe
        if len(subreddits_to_ban) > 0:
            modified_df = modified_df[~modified_df['subreddit'].isin(subreddits_to_ban)]

            # Remove corresponding nodes from graph
            content_to_remove = self.df[self.df['subreddit'].isin(subreddits_to_ban)]['id'].tolist()
            post_to_id = getattr(self.graph_data, 'post_to_id', {})
            nodes_to_remove = []

            for content_id in content_to_remove:
                if content_id in post_to_id:
                    node_id = post_to_id[content_id]
                    if modified_graph.has_node(node_id):
                        nodes_to_remove.append(node_id)

            modified_graph.remove_nodes_from(nodes_to_remove)
        else:
            nodes_to_remove = []

        # Calculate impact
        original_hate_ratio = len(self.hate_content) / len(self.df)
        remaining_hate = modified_df[modified_df['davidson_label'] == 2] if len(modified_df) > 0 else pd.DataFrame()
        new_hate_ratio = len(remaining_hate) / len(modified_df) if len(modified_df) > 0 else 0

        impact = {
            'strategy': 'subreddit_banning',
            'ban_percentage': ban_percentage,
            'subreddits_banned': num_to_ban,
            'banned_subreddits': list(subreddits_to_ban),
            'nodes_removed': len(nodes_to_remove),
            'original_hate_ratio': original_hate_ratio,
            'new_hate_ratio': new_hate_ratio,
            'hate_reduction': (original_hate_ratio - new_hate_ratio) / original_hate_ratio if original_hate_ratio > 0 else 0,
            'graph_connectivity': self.calculate_connectivity(modified_graph),
            'remaining_samples': len(modified_df)
        }

        return impact, modified_graph, modified_df

    def simulate_combined_intervention(self, intervention_percentage=0.15):
        """Simulate combined moderation intervention."""
        print(f"Simulating combined intervention ({intervention_percentage*100}%)...")

        # Apply multiple strategies with reduced individual impact
        content_impact, graph1, df1 = self.simulate_content_removal(intervention_percentage * 0.6)

        # Update hate users and subreddits based on remaining data
        self.update_hate_identification(df1)

        user_impact, graph2, df2 = self.simulate_user_banning(intervention_percentage * 0.3)
        subreddit_impact, graph3, df3 = self.simulate_subreddit_banning(intervention_percentage * 0.1)

        # Calculate combined impact
        original_hate_ratio = len(self.hate_content) / len(self.df)
        final_hate_ratio = content_impact['new_hate_ratio']  # Use the most comprehensive result

        combined_impact = {
            'strategy': 'combined_intervention',
            'intervention_percentage': intervention_percentage,
            'content_removed': content_impact['content_removed'],
            'users_banned': user_impact['users_banned'],
            'subreddits_banned': subreddit_impact['subreddits_banned'],
            'original_hate_ratio': original_hate_ratio,
            'new_hate_ratio': final_hate_ratio,
            'hate_reduction': content_impact['hate_reduction'],
            'graph_connectivity': self.calculate_connectivity(graph1),
            'remaining_samples': content_impact['remaining_samples']
        }

        return combined_impact, graph1, df1

    def update_hate_identification(self, df):
        """Update hate identification based on remaining data."""
        # Re-identify hate users from remaining data
        hate_users = []
        for user, group in df.groupby('author'):
            if pd.isna(user):
                continue
            total_posts = len(group)
            hate_posts = len(group[group['davidson_label'] == 2])
            hate_ratio = hate_posts / total_posts if total_posts > 0 else 0

            if hate_ratio > self.hate_threshold and total_posts >= 3:
                hate_users.append({
                    'user': user,
                    'hate_ratio': hate_ratio,
                    'total_posts': total_posts,
                    'hate_posts': hate_posts
                })

        self.hate_users = hate_users

    def calculate_connectivity(self, graph):
        """Calculate graph connectivity metrics."""
        if graph.number_of_nodes() == 0:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'avg_clustering': 0,
                'num_components': 0,
                'largest_component_size': 0
            }

        try:
            # Basic metrics
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()

            # Clustering coefficient
            avg_clustering = nx.average_clustering(graph) if num_nodes > 0 else 0

            # Connected components
            components = list(nx.connected_components(graph))
            num_components = len(components)
            largest_component_size = len(max(components, key=len)) if components else 0

            return {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'avg_clustering': avg_clustering,
                'num_components': num_components,
                'largest_component_size': largest_component_size
            }
        except:
            return {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'avg_clustering': 0,
                'num_components': 0,
                'largest_component_size': 0
            }

    def evaluate_moderation_effectiveness(self, impacts):
        """Evaluate overall effectiveness of moderation strategies."""
        print("Evaluating moderation effectiveness...")

        effectiveness_scores = {}

        for strategy, impact in impacts.items():
            # Effectiveness score based on multiple factors
            hate_reduction_score = impact.get('hate_reduction', 0) * 100

            # Penalty for reducing graph connectivity too much
            connectivity = impact.get('graph_connectivity', {})
            connectivity_penalty = 0

            original_connectivity = self.calculate_connectivity(self.original_graph)
            if original_connectivity['num_nodes'] > 0:
                node_reduction = 1 - (connectivity['num_nodes'] / original_connectivity['num_nodes'])
                connectivity_penalty = node_reduction * 20  # Penalty for removing too many nodes

            # Penalty for removing too much content
            content_penalty = 0
            if 'remaining_samples' in impact:
                content_reduction = 1 - (impact['remaining_samples'] / len(self.df))
                if content_reduction > 0.5:  # Penalty if removing more than 50% of content
                    content_penalty = (content_reduction - 0.5) * 40

            effectiveness_score = max(0, hate_reduction_score - connectivity_penalty - content_penalty)
            effectiveness_scores[strategy] = effectiveness_score

        return effectiveness_scores

    def run_moderation_simulation(self):
        """Run complete moderation simulation."""
        print("Running moderation simulation...")

        # Identify hate content
        identification_stats = self.identify_hate_content()

        # Simulate different strategies
        strategies = {}

        # Content removal
        content_impact, _, _ = self.simulate_content_removal(
            self.moderation_config.get('content_removal_percentage', 0.2)
        )
        strategies['content_removal'] = content_impact

        # User banning
        user_impact, _, _ = self.simulate_user_banning(
            self.moderation_config.get('ban_percentage', 0.1)
        )
        strategies['user_banning'] = user_impact

        # Subreddit banning
        subreddit_impact, _, _ = self.simulate_subreddit_banning(
            self.moderation_config.get('subreddit_ban_percentage', 0.1)
        )
        strategies['subreddit_banning'] = subreddit_impact

        # Combined intervention
        combined_impact, _, _ = self.simulate_combined_intervention(
            self.moderation_config.get('intervention_percentage', 0.15)
        )
        strategies['combined_intervention'] = combined_impact

        # Evaluate effectiveness
        effectiveness_scores = self.evaluate_moderation_effectiveness(strategies)

        results = {
            'identification_stats': identification_stats,
            'strategy_impacts': strategies,
            'effectiveness_scores': effectiveness_scores,
            'original_graph_stats': self.calculate_connectivity(self.original_graph)
        }

        return results

def save_moderation_results(results, config):
    """Save moderation simulation results."""
    print("Saving moderation simulation results...")

    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)

    # Save complete results
    with open(artifacts_dir / 'moderation_simulation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save effectiveness summary
    effectiveness_summary = {
        'best_strategy': max(results['effectiveness_scores'], key=results['effectiveness_scores'].get),
        'effectiveness_scores': results['effectiveness_scores'],
        'hate_reduction_by_strategy': {
            strategy: impact.get('hate_reduction', 0)
            for strategy, impact in results['strategy_impacts'].items()
        }
    }

    with open(artifacts_dir / 'moderation_effectiveness.json', 'w') as f:
        json.dump(effectiveness_summary, f, indent=2)

    print(f"Moderation simulation results saved to {artifacts_dir}")

def main():
    parser = argparse.ArgumentParser(description="Moderation strategy simulation for Reddit hate speech")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    print("=== Moderation Strategy Simulation ===")

    # Initialize simulator
    simulator = ModerationSimulator(config)

    # Load data
    simulator.load_data()

    # Run simulation
    results = simulator.run_moderation_simulation()

    # Save results
    save_moderation_results(results, config)

    print(f"\n=== Moderation Simulation Summary ===")
    print(f"Hate content identified: {results['identification_stats']['hate_content_count']}")
    print(f"Hate users identified: {results['identification_stats']['hate_users_count']}")
    print(f"Hate subreddits identified: {results['identification_stats']['hate_subreddits_count']}")

    print(f"\nStrategy Effectiveness Scores:")
    for strategy, score in results['effectiveness_scores'].items():
        print(f"  {strategy}: {score:.2f}")

    print(f"\nBest Strategy: {max(results['effectiveness_scores'], key=results['effectiveness_scores'].get)}")

    print(f"\nHate Reduction by Strategy:")
    for strategy, impact in results['strategy_impacts'].items():
        print(f"  {strategy}: {impact.get('hate_reduction', 0)*100:.1f}%")

if __name__ == "__main__":
    main()
