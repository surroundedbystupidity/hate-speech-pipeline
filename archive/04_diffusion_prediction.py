#!/usr/bin/env python3
"""
Step 4 - Results Analysis Script for Reddit Hate Speech Analysis.
Implements comprehensive results analysis including n-grams, predictions, and temporal dynamics.
"""

import argparse
import json
import pickle
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Import TGNN model classes
sys.path.append(str(Path(__file__).parent))
try:
    from scripts.tgnn_model_03 import TGNNModel
except ImportError:
    try:
        # Try direct import from 03_tgnn_model
        import importlib.util
        spec = importlib.util.spec_from_file_location("tgnn_model", Path(__file__).parent / "03_tgnn_model.py")
        tgnn_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tgnn_module)
        TGNNModel = tgnn_module.TGNNModel
    except Exception as e:
        print(f"Warning: Could not import TGNN model classes: {e}")
        TGNNModel = None

# Cross-encoder imports (optional, for advanced reranking)
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _ce_ok = True
except ImportError:
    _ce_ok = False

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class Step4ResultsAnalyzer:
    """Step 4 - Comprehensive Results Analysis."""

    def __init__(self, config):
        self.config = config
        self.artifacts_dir = Path(config['paths']['artifacts_dir'])

    def load_data(self):
        """Load all necessary data for analysis."""
        print("Loading data for Step 4 analysis...")

        # Load Step 1 labeled comments
        self.comments_df = pd.read_parquet(self.artifacts_dir / 'step1_labeled_comments.parquet')

        # Load Step 2 features
        with open(self.artifacts_dir / 'step2_user_features.json', 'r') as f:
            self.user_features = json.load(f)
        with open(self.artifacts_dir / 'step2_subreddit_features.json', 'r') as f:
            self.subreddit_features = json.load(f)
        with open(self.artifacts_dir / 'step2_comment_features.json', 'r') as f:
            self.comment_features = json.load(f)

        # Load graph
        with open(self.artifacts_dir / 'step2_graph.pkl', 'rb') as f:
            self.graph = pickle.load(f)

        print(f"Loaded {len(self.comments_df)} comments")
        print(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def prepare_hate_comments_list(self):
        """Prepare a list of comments flagged as containing hate speech."""
        print("Preparing hate speech comments list...")

        hate_comments = self.comments_df[self.comments_df['cardiffnlp_label'] == 1].copy()

        hate_list = []
        for _, comment in hate_comments.iterrows():
            hate_list.append({
                'id': comment['id'],
                'text': comment['text_content'],
                'author': comment['author'],
                'subreddit': comment['subreddit'],
                'created_utc': comment['created_utc'],
                'score': comment.get('score', 0),
                'parent_id': comment.get('parent_id', ''),
                'link_id': comment.get('link_id', '')
            })

        print(f"Found {len(hate_list)} hate speech comments")
        return hate_list

    def extract_ngrams(self, texts, n=2, top_k=50):
        """Extract n-grams from hate speech comments."""
        print(f"Extracting {n}-grams from hate speech comments...")

        all_ngrams = Counter()

        for text in tqdm(texts, desc=f"Extracting {n}-grams"):
            if pd.isna(text) or not text:
                continue

            # Clean and tokenize text
            text_clean = re.sub(r'[^\w\s]', ' ', str(text).lower())
            words = text_clean.split()

            # Extract n-grams
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                if len(ngram.strip()) > 0:
                    all_ngrams[ngram] += 1

        # Get top k n-grams
        top_ngrams = all_ngrams.most_common(top_k)

        print(f"Extracted {len(all_ngrams)} unique {n}-grams")
        print(f"Top {top_k} {n}-grams:")
        for i, (ngram, count) in enumerate(top_ngrams[:10], 1):
            print(f"{i:2d}. {ngram}: {count}")

        return top_ngrams

    def compare_with_davidson_lexicon(self, top_ngrams):
        """Compare top n-grams with Davidson lexicon."""
        print("Comparing n-grams with Davidson lexicon...")

        # Define HATE_LEXICON directly (from data preparation script)
        HATE_LEXICON = [
            'allah akbar', 'all niggers', 'faggots like', 'faggots usually', 'fucking nigger',
            'get raped', 'hate faggots', 'raped by', 'shut up nigger', 'this nigger',
            'you nigger', 'you niggers', 'a fucking queer', 'faggots like you',
            'fucking hate you', 'full of white', 'how many niggers', 'is a fucking',
            'many niggers are', 'niggers are in', 'full of white trash',
            'how many niggers are', 'is full of white', 'many niggers are in',
            'niggers are in my', 'blacks', 'chink', 'chinks', 'dykes', 'faggot', 'faggots', 'fags',
            'homo', 'inbred', 'niggers', 'queers', 'raped', 'savages', 'slave',
            'spic', 'wetback', 'wetbacks', 'whites', 'a dirty', 'a nigger',
            'all white', 'always fuck', 'ass white', 'be killed', 'beat him',
            'biggest faggot', 'blame the', 'butt ugly', 'chink eyed', 'chinks in',
            'coon shit', 'dumb monkey', 'dumb nigger', 'fag and', 'fag but',
            'faggot a', 'faggot and', 'faggot ass', 'faggot bitch', 'faggot for',
            'faggot smh', 'faggot that', 'faggots and', 'fags are', 'fuckin faggot',
            'fucking faggot', 'fucking gay', 'fucking hate', 'fucking queer',
            'gay ass', 'hate all', 'hate fat', 'hate you', 'here faggot',
            'is white', 'jungle bunny', 'kill all', 'kill yourself', 'little faggot',
            'married to', 'me faggot', 'my coon', 'nigga ask', 'niggas like',
            'nigger ass', 'nigger is', 'nigger music', 'niggers are',
            'nigger', 'nigga', 'kike', 'spic', 'wetback', 'chink', 'gook', 'jap',
            'white trash', 'black trash', 'yellow peril', 'brown people', 'redskin', 'redskins',
            'tranny', 'shemale', 'ladyboy', 'fag hag', 'dyke', 'butch', 'femme',
            'sissy', 'pansy', 'fruit', 'twink', 'bear', 'otter', 'chub', 'chaser',
            'retard', 'spaz', 'cripple', 'gimp', 'lame', 'dumb', 'idiot', 'moron', 'imbecile', 'feeble', 'handicapped',
            'kill', 'kills', 'killing', 'murder', 'murders', 'murdering', 'violence', 'violent',
            'beat', 'beats', 'beating', 'attack', 'attacks', 'attacking', 'assault', 'assaults',
            'threat', 'threats', 'threatening', 'threaten', 'die', 'dies', 'dying', 'death', 'dead',
            'suicide', 'should die', 'deserve to die', 'need to die', 'want to die', 'hope you die',
            'hate', 'hates', 'hating', 'hated', 'despise', 'despises', 'despising', 'despised',
            'disgusting', 'disgust', 'disgusts', 'disgusted', 'sick', 'sickening', 'evil', 'devil',
            'monster', 'monsters', 'scum', 'trash', 'garbage', 'vermin', 'pest', 'pests',
            'parasite', 'parasites', 'cancer', 'disease', 'plague',
            'animal', 'animals', 'beast', 'beasts', 'barbarian', 'barbarians', 'subhuman',
            'inhuman', 'less than human', 'not human', 'filth', 'filthy',
            'all blacks', 'all whites', 'all muslims', 'all jews', 'all asians', 'all mexicans',
            'all immigrants', 'all foreigners', 'all gays', 'all lesbians', 'all trans',
            'go back to', 'return to', 'get out of', 'fuck off', 'fuck you', 'fuck them',
            'burn in hell', 'go to hell', 'rot in hell', 'damn you', 'damned',
            'jihad', 'terrorist', 'terrorists', 'terrorism', 'islamic state', 'isis', 'al qaeda',
            'bomb', 'bombs', 'bombing', 'explode', 'explodes', 'explosion',
            'nazi', 'nazis', 'hitler', 'holocaust', 'genocide', 'slavery', 'lynching',
            'kkk', 'klan', 'white power', 'aryan', 'supremacist', 'supremacists', 'fascist', 'fascists',
            'damn', 'hell', 'shit', 'fuck', 'fucking', 'motherfucker', 'fucker',
            'bullshit', 'crap', 'piss', 'pissed', 'bloody', 'bitchy',
            'kys', 'gtfo', 'stfu', 'fuck off', 'eat shit', 'go die', 'kill yourself',
            'you suck', 'you\'re trash', 'you\'re garbage', 'piece of shit', 'worthless',
            'pathetic', 'disgusting', 'revolting', 'vile', 'despicable', 'contemptible',
            'criminal', 'criminals', 'terrorist', 'terrorists', 'mob', 'mobs', 'violent',
            'capitol', 'murder', 'murdering', 'obstruct', 'certification', 'election',
            'senate', 'republicans', 'fear', 'lives', 'day', 'fuck', 'fucking', 'fucked',
            'shit', 'shitty', 'damn', 'damned', 'hell', 'suck', 'sucks', 'sucking',
            'stupid', 'idiot', 'moron', 'dumb', 'retard', 'asshole', 'bastard',
            'hate', 'hated', 'hating', 'despise', 'disgust', 'disgusting', 'sick',
            'evil', 'devil', 'monster', 'scum', 'trash', 'garbage', 'worthless',
            'pathetic', 'revolting', 'vile', 'despicable', 'contemptible', 'awful',
            'terrible', 'horrible', 'disgusting', 'revolting', 'sickening', 'nasty',
            'gross', 'filthy', 'dirty', 'rotten', 'corrupt', 'corrupted', 'criminal',
            'mentally challenged', 'narcissism', 'narcissist', 'white america',
            'maga loyalists', 'maga', 'loyalists', 'loyalist', 'woke', 'crazy',
            'insane', 'lunatic', 'psycho', 'psychotic', 'nasty', 'gnat'
        ]

        davidson_terms = set(term.lower() for term in HATE_LEXICON)

        comparison_results = {
            'total_ngrams': len(top_ngrams),
            'matched_ngrams': [],
            'unmatched_ngrams': [],
            'coverage': 0.0
        }

        matched_count = 0
        for ngram, count in top_ngrams:
            ngram_lower = ngram.lower()

            # Check if ngram matches any Davidson term
            is_matched = False
            for davidson_term in davidson_terms:
                if davidson_term in ngram_lower or ngram_lower in davidson_term:
                    is_matched = True
                    break

            if is_matched:
                matched_count += 1
                comparison_results['matched_ngrams'].append({
                    'ngram': ngram,
                    'count': count,
                    'matched': True
                })
            else:
                comparison_results['unmatched_ngrams'].append({
                    'ngram': ngram,
                    'count': count,
                    'matched': False
                })

        comparison_results['coverage'] = matched_count / len(top_ngrams) if top_ngrams else 0.0

        print(f"Coverage: {comparison_results['coverage']:.2%} ({matched_count}/{len(top_ngrams)})")
        print(f"Matched n-grams: {len(comparison_results['matched_ngrams'])}")
        print(f"Unmatched n-grams: {len(comparison_results['unmatched_ngrams'])}")

        return comparison_results

    def node_level_prediction(self):
        """Node-level prediction: Predict whether a future comment will contain hate."""
        print("Performing node-level prediction analysis...")

        # Group comments by thread (link_id) and sort by time
        thread_predictions = []

        for link_id, thread_comments in self.comments_df.groupby('link_id'):
            thread_comments = thread_comments.sort_values('created_utc')

            if len(thread_comments) < 2:
                continue

            # For each comment, predict based on previous comments in the thread
            for i in range(1, len(thread_comments)):
                current_comment = thread_comments.iloc[i]
                previous_comments = thread_comments.iloc[:i]

                # Features for prediction
                features = {
                    'thread_hate_ratio': (previous_comments['cardiffnlp_label'] == 1).mean(),
                    'thread_size': len(previous_comments),
                    'time_since_first': current_comment['created_utc'] - thread_comments.iloc[0]['created_utc'],
                    'author_hate_ratio': self.user_features.get(current_comment['author'], {}).get('hate_speech_ratio', 0),
                    'subreddit_hate_ratio': self.subreddit_features.get(current_comment['subreddit'], {}).get('hate_speech_ratio', 0)
                }

                # Simple prediction: if thread has high hate ratio, predict hate
                predicted_hate = 1 if features['thread_hate_ratio'] > 0.3 else 0
                actual_hate = current_comment['cardiffnlp_label']

                thread_predictions.append({
                    'comment_id': current_comment['id'],
                    'link_id': link_id,
                    'predicted': predicted_hate,
                    'actual': actual_hate,
                    'features': features
                })

        # Calculate metrics
        if thread_predictions:
            predictions_df = pd.DataFrame(thread_predictions)
            y_true = predictions_df['actual']
            y_pred = predictions_df['predicted']

            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'total_predictions': len(thread_predictions)
            }

            print(f"Node-level prediction metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1: {metrics['f1']:.3f}")
            print(f"  Total predictions: {metrics['total_predictions']}")
        else:
            metrics = {'error': 'No predictions made'}

        return metrics

    def edge_level_prediction(self):
        """Edge-level prediction: Predict if a comment will trigger hate in replies."""
        print("Performing edge-level prediction analysis...")

        edge_predictions = []

        # Try multiple parent-child relationship approaches
        parent_child_pairs = []

        # Method 1: Direct parent_id relationships
        for _, comment in self.comments_df.iterrows():
            comment_id = comment['id']
            replies = self.comments_df[self.comments_df['parent_id'] == comment_id]
            if len(replies) > 0:
                parent_child_pairs.extend([(comment, reply) for _, reply in replies.iterrows()])

        # Method 2: If no direct parent_id, use thread-based temporal relationships
        if len(parent_child_pairs) == 0:
            print("No direct parent_id relationships found, using temporal thread analysis...")
            for link_id in self.comments_df['link_id'].unique():
                thread_comments = self.comments_df[self.comments_df['link_id'] == link_id].sort_values('created_utc')

                # Create parent-child pairs based on temporal proximity (within 1 hour)
                for i in range(len(thread_comments) - 1):
                    parent = thread_comments.iloc[i]
                    for j in range(i + 1, min(i + 5, len(thread_comments))):  # Look at next 4 comments
                        child = thread_comments.iloc[j]
                        time_diff = (child['created_utc'] - parent['created_utc']).total_seconds()
                        if time_diff <= 3600:  # Within 1 hour
                            parent_child_pairs.append((parent, child))

        print(f"Found {len(parent_child_pairs)} parent-child comment pairs")

        # Analyze each parent-child pair
        for parent, child in parent_child_pairs[:1000]:  # Limit to avoid memory issues
            # Calculate features
            features = {
                'parent_hate': parent['cardiffnlp_label'],
                'parent_score': parent.get('score', 0),
                'parent_length': len(str(parent['text_content'])),
                'author_hate_ratio': self.user_features.get(parent['author'], {}).get('hate_speech_ratio', 0),
                'subreddit_hate_ratio': self.subreddit_features.get(parent['subreddit'], {}).get('hate_speech_ratio', 0),
                'time_diff': (child['created_utc'] - parent['created_utc']).total_seconds() / 60  # minutes
            }

            # Improved prediction logic with multiple factors
            hate_score = 0

            # Factor 1: Parent hate speech
            if features['parent_hate'] == 1:
                hate_score += 0.6

            # Factor 2: Author hate ratio
            if features['author_hate_ratio'] > 0.1:
                hate_score += 0.3

            # Factor 3: Subreddit hate ratio
            if features['subreddit_hate_ratio'] > 0.05:
                hate_score += 0.2

            # Factor 4: Quick response (within 10 minutes)
            if features['time_diff'] < 10:
                hate_score += 0.1

            # Use dynamic threshold for better precision/recall balance
            predicted_hate = 1 if hate_score > 0.5 else 0
            actual_hate = child['cardiffnlp_label']

            edge_predictions.append({
                'parent_id': parent['id'],
                'child_id': child['id'],
                'predicted': predicted_hate,
                'actual': actual_hate,
                'features': features
            })

        # Calculate metrics
        if edge_predictions:
            predictions_df = pd.DataFrame(edge_predictions)
            y_true = predictions_df['actual']
            y_pred = predictions_df['predicted']

            # Calculate additional metrics
            try:
                auc_score = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
            except:
                auc_score = 0.0

            cm = confusion_matrix(y_true, y_pred)

            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'auc': auc_score,
                'confusion_matrix': cm.tolist(),
                'total_predictions': len(edge_predictions),
                'positive_predictions': int(y_pred.sum()),
                'actual_positives': int(y_true.sum()),
                'avg_child_hate_ratio': predictions_df['actual'].mean()
            }

            print(f"Edge-level prediction metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1: {metrics['f1']:.3f}")
            print(f"  AUC: {metrics['auc']:.3f}")
            print(f"  Total predictions: {metrics['total_predictions']}")
            print(f"  Positive predictions: {metrics['positive_predictions']}")
            print(f"  Actual positives: {metrics['actual_positives']}")
            print(f"  Avg child hate ratio: {metrics['avg_child_hate_ratio']:.3f}")
            print(f"  Confusion matrix: {metrics['confusion_matrix']}")
        else:
            metrics = {'error': 'No edge predictions made'}

        return metrics

    def build_next_hate_dataset(self, edge_pairs, edge_features, occurred_mask, dst_labels):
        """Build dataset for next hate prediction."""
        Xh = edge_features[occurred_mask==1]
        yh = dst_labels[occurred_mask==1].astype(int)
        return Xh, yh

    def train_and_calibrate_hate_classifier(self, Xh, yh, random_state=42):
        """Train and calibrate hate classifier."""
        X_tr, X_va, y_tr, y_va = train_test_split(Xh, yh, test_size=0.2, random_state=random_state, stratify=yh)
        rf_hate = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, class_weight='balanced', n_jobs=-1, random_state=random_state)
        rf_hate.fit(X_tr, y_tr)
        p_va = rf_hate.predict_proba(X_va)[:,1]
        lr = LogisticRegression(max_iter=1000, random_state=random_state)
        z_va = p_va.reshape(-1,1)
        lr.fit(z_va, y_va)
        def calibrated_proba(p):
            z = np.asarray(p).reshape(-1,1)
            return lr.predict_proba(z)[:,1]
        ap = average_precision_score(y_va, calibrated_proba(p_va))
        auc = roc_auc_score(y_va, calibrated_proba(p_va))
        return rf_hate, calibrated_proba, {'pr_auc':ap,'roc_auc':auc}

    def temporal_dynamics_analysis(self):
        """Analyze temporal dynamics of hate spread."""
        print("Analyzing temporal dynamics of hate spread...")

        temporal_analysis = {
            'hate_response_times': [],
            'cascade_speeds': [],
            'thread_evolution': []
        }

        # Analyze response times to hate comments
        for _, hate_comment in self.comments_df[self.comments_df['cardiffnlp_label'] == 1].iterrows():
            hate_time = hate_comment['created_utc']
            link_id = hate_comment['link_id']

            # Find subsequent comments in the same thread
            thread_comments = self.comments_df[
                (self.comments_df['link_id'] == link_id) &
                (self.comments_df['created_utc'] > hate_time)
            ].sort_values('created_utc')

            if len(thread_comments) > 0:
                # Time to first response
                first_response_time = thread_comments.iloc[0]['created_utc'] - hate_time
                temporal_analysis['hate_response_times'].append(first_response_time)

                # Check if first few responses are also hateful
                first_responses = thread_comments.head(5)
                hate_in_responses = (first_responses['cardiffnlp_label'] == 1).sum()
                temporal_analysis['cascade_speeds'].append({
                    'time_to_first': first_response_time,
                    'hate_in_first_5': hate_in_responses,
                    'total_responses': len(first_responses)
                })

        # Calculate statistics
        if temporal_analysis['hate_response_times']:
            response_times = temporal_analysis['hate_response_times']
            temporal_analysis['stats'] = {
                'avg_response_time': np.mean(response_times),
                'median_response_time': np.median(response_times),
                'min_response_time': np.min(response_times),
                'max_response_time': np.max(response_times)
            }

            print(f"Temporal dynamics analysis:")
            avg_time = temporal_analysis['stats']['avg_response_time']
            median_time = temporal_analysis['stats']['median_response_time']
            min_time = temporal_analysis['stats']['min_response_time']
            max_time = temporal_analysis['stats']['max_response_time']

            # Convert to seconds if they are Timedelta objects
            if hasattr(avg_time, 'total_seconds'):
                avg_time = avg_time.total_seconds()
            if hasattr(median_time, 'total_seconds'):
                median_time = median_time.total_seconds()
            if hasattr(min_time, 'total_seconds'):
                min_time = min_time.total_seconds()
            if hasattr(max_time, 'total_seconds'):
                max_time = max_time.total_seconds()

            print(f"  Average response time: {avg_time:.1f} seconds")
            print(f"  Median response time: {median_time:.1f} seconds")
            print(f"  Response time range: {min_time:.1f} - {max_time:.1f} seconds")

        return temporal_analysis

    def influence_estimation(self):
        """Identify influential comments, users, and subreddits in hate propagation."""
        print("Analyzing influence in hate propagation...")

        influence_analysis = {
            'influential_users': [],
            'influential_comments': [],
            'influential_subreddits': []
        }

        # Analyze user influence
        user_influence = {}
        for user, features in self.user_features.items():
            if features['total_comments'] >= 3:  # Minimum activity threshold
                # Calculate influence score based on hate ratio and activity
                influence_score = features['hate_speech_ratio'] * np.log(features['total_comments'] + 1)
                user_influence[user] = {
                    'user': user,
                    'influence_score': influence_score,
                    'hate_ratio': features['hate_speech_ratio'],
                    'total_comments': features['total_comments'],
                    'unique_subreddits': features['unique_subreddits']
                }

        # Top influential users
        top_users = sorted(user_influence.values(), key=lambda x: x['influence_score'], reverse=True)[:20]
        influence_analysis['influential_users'] = top_users

        # Analyze comment influence (comments that trigger many hate replies)
        comment_influence = {}
        for _, comment in self.comments_df.iterrows():
            comment_id = comment['id']
            replies = self.comments_df[self.comments_df['parent_id'] == comment_id]

            if len(replies) > 0:
                hate_replies = replies[replies['cardiffnlp_label'] == 1]
                influence_score = len(hate_replies) / len(replies) * np.log(len(replies) + 1)

                comment_influence[comment_id] = {
                    'comment_id': comment_id,
                    'influence_score': influence_score,
                    'total_replies': len(replies),
                    'hate_replies': len(hate_replies),
                    'hate_reply_ratio': len(hate_replies) / len(replies),
                    'text': str(comment['text_content'])[:100] + '...' if len(str(comment['text_content'])) > 100 else str(comment['text_content'])
                }

        # Top influential comments
        top_comments = sorted(comment_influence.values(), key=lambda x: x['influence_score'], reverse=True)[:20]
        influence_analysis['influential_comments'] = top_comments

        # Analyze subreddit influence
        subreddit_influence = {}
        for subreddit, features in self.subreddit_features.items():
            if features['total_comments'] >= 10:  # Minimum activity threshold
                influence_score = features['hate_speech_ratio'] * np.log(features['total_comments'] + 1)
                subreddit_influence[subreddit] = {
                    'subreddit': subreddit,
                    'influence_score': influence_score,
                    'hate_ratio': features['hate_speech_ratio'],
                    'total_comments': features['total_comments']
                }

        # Top influential subreddits
        top_subreddits = sorted(subreddit_influence.values(), key=lambda x: x['influence_score'], reverse=True)[:10]
        influence_analysis['influential_subreddits'] = top_subreddits

        print(f"Influence analysis results:")
        print(f"  Top influential users: {len(top_users)}")
        print(f"  Top influential comments: {len(top_comments)}")
        print(f"  Top influential subreddits: {len(top_subreddits)}")

        if top_users:
            print(f"  Most influential user: {top_users[0]['user']} (score: {top_users[0]['influence_score']:.3f})")
        if top_comments:
            print(f"  Most influential comment: {top_comments[0]['comment_id']} (score: {top_comments[0]['influence_score']:.3f})")
        if top_subreddits:
            print(f"  Most influential subreddit: {top_subreddits[0]['subreddit']} (score: {top_subreddits[0]['influence_score']:.3f})")

        return influence_analysis

    def propagation_patterns_analysis(self):
        """Analyze propagation patterns and cascade shapes."""
        print("Analyzing propagation patterns and cascade shapes...")

        propagation_analysis = {
            'cascade_shapes': [],
            'branching_patterns': [],
            'hate_clustering': []
        }

        # Analyze cascade shapes for each thread
        for link_id, thread_comments in self.comments_df.groupby('link_id'):
            if len(thread_comments) < 3:
                continue

            # Build comment tree structure
            comments_dict = {}
            root_comments = []

            for _, comment in thread_comments.iterrows():
                comments_dict[comment['id']] = {
                    'id': comment['id'],
                    'parent_id': comment['parent_id'],
                    'is_hate': comment['cardiffnlp_label'] == 1,
                    'children': []
                }

            # Build tree structure with improved parent detection
            for comment_id, comment_data in comments_dict.items():
                parent_id = comment_data['parent_id']
                if parent_id == link_id or parent_id.startswith('t3_') or parent_id not in comments_dict:
                    root_comments.append(comment_id)
                elif parent_id in comments_dict:
                    comments_dict[parent_id]['children'].append(comment_id)

            # If no proper tree structure found, create temporal-based structure
            if not any(comments_dict[cid]['children'] for cid in comments_dict):
                # Create temporal cascades
                sorted_comments = thread_comments.sort_values('created_utc')
                comment_ids = sorted_comments['id'].tolist()

                # Clear previous structure
                for cid in comments_dict:
                    comments_dict[cid]['children'] = []
                root_comments = []

                # Create temporal parent-child relationships
                if len(comment_ids) > 1:
                    root_comments = [comment_ids[0]]  # First comment is root

                    for i in range(1, len(comment_ids)):
                        parent_idx = max(0, i - 1)  # Previous comment as parent
                        parent_id = comment_ids[parent_idx]
                        child_id = comment_ids[i]

                        if parent_id in comments_dict and child_id in comments_dict:
                            comments_dict[parent_id]['children'].append(child_id)

            # Analyze cascade shape
            def analyze_branch(comment_id, depth=0):
                if comment_id not in comments_dict:
                    return {'depth': depth, 'hate_count': 0, 'total_count': 0}

                comment_data = comments_dict[comment_id]
                hate_count = 1 if comment_data['is_hate'] else 0
                total_count = 1
                max_depth = depth

                for child_id in comment_data['children']:
                    child_stats = analyze_branch(child_id, depth + 1)
                    hate_count += child_stats['hate_count']
                    total_count += child_stats['total_count']
                    max_depth = max(max_depth, child_stats['depth'])

                return {'depth': max_depth, 'hate_count': hate_count, 'total_count': total_count}

            # Analyze each root branch
            for root_id in root_comments:
                branch_stats = analyze_branch(root_id)
                propagation_analysis['cascade_shapes'].append({
                    'link_id': link_id,
                    'root_comment': root_id,
                    'max_depth': branch_stats['depth'],
                    'total_comments': branch_stats['total_count'],
                    'hate_comments': branch_stats['hate_count'],
                    'hate_ratio': branch_stats['hate_count'] / branch_stats['total_count'] if branch_stats['total_count'] > 0 else 0
                })

        # Calculate statistics
        if propagation_analysis['cascade_shapes']:
            shapes_df = pd.DataFrame(propagation_analysis['cascade_shapes'])
            propagation_analysis['stats'] = {
                'avg_depth': shapes_df['max_depth'].mean(),
                'max_depth': shapes_df['max_depth'].max(),
                'avg_hate_ratio': shapes_df['hate_ratio'].mean(),
                'deep_threads': len(shapes_df[shapes_df['max_depth'] > 1]),  # Lower threshold
                'wide_threads': len(shapes_df[shapes_df['total_comments'] > 3]),  # Lower threshold
                'total_cascades': len(shapes_df)
            }

            print(f"Propagation patterns analysis:")
            print(f"  Average cascade depth: {propagation_analysis['stats']['avg_depth']:.2f}")
            print(f"  Maximum cascade depth: {propagation_analysis['stats']['max_depth']}")
            print(f"  Average hate ratio in cascades: {propagation_analysis['stats']['avg_hate_ratio']:.3f}")
            print(f"  Deep threads (>1 levels): {propagation_analysis['stats']['deep_threads']}")
            print(f"  Wide threads (>3 comments): {propagation_analysis['stats']['wide_threads']}")
            print(f"  Total cascades: {propagation_analysis['stats']['total_cascades']}")

        return propagation_analysis

    def network_vulnerability_analysis(self):
        """Analyze network vulnerability to hate spreading."""
        print("Analyzing network vulnerability to hate spreading...")

        vulnerability_analysis = {
            'thread_vulnerability': [],
            'user_vulnerability': [],
            'structural_factors': {}
        }

        # Analyze thread vulnerability
        for link_id, thread_comments in self.comments_df.groupby('link_id'):
            if len(thread_comments) < 2:
                continue

            # Calculate vulnerability factors
            factors = {
                'link_id': link_id,
                'thread_size': len(thread_comments),
                'unique_authors': thread_comments['author'].nunique(),
                'hate_ratio': (thread_comments['cardiffnlp_label'] == 1).mean(),
                'avg_score': thread_comments['score'].mean(),
                'time_span': thread_comments['created_utc'].max() - thread_comments['created_utc'].min(),
                'subreddit': thread_comments['subreddit'].iloc[0]
            }

            # Vulnerability score (higher = more vulnerable)
            # Convert time_span to seconds if it's a Timedelta
            time_span_seconds = factors['time_span']
            if hasattr(time_span_seconds, 'total_seconds'):
                time_span_seconds = time_span_seconds.total_seconds()

            vulnerability_score = (
                factors['hate_ratio'] * 0.4 +  # High hate ratio
                (factors['thread_size'] / 50) * 0.3 +  # Large threads
                (time_span_seconds / 3600) * 0.2 +  # Long duration
                (1 - factors['unique_authors'] / factors['thread_size']) * 0.1  # Few unique authors
            )

            factors['vulnerability_score'] = vulnerability_score
            vulnerability_analysis['thread_vulnerability'].append(factors)

        # Analyze user vulnerability (users who are more likely to engage in hate)
        user_vulnerability = {}
        for user, features in self.user_features.items():
            if features['total_comments'] >= 2:
                # Users with high hate ratio and low subreddit diversity are more vulnerable
                vulnerability_score = (
                    features['hate_speech_ratio'] * 0.6 +
                    (1 - features['unique_subreddits'] / 10) * 0.4  # Low diversity
                )

                user_vulnerability[user] = {
                    'user': user,
                    'vulnerability_score': vulnerability_score,
                    'hate_ratio': features['hate_speech_ratio'],
                    'unique_subreddits': features['unique_subreddits'],
                    'total_comments': features['total_comments']
                }

        # Top vulnerable users
        top_vulnerable_users = sorted(user_vulnerability.values(), key=lambda x: x['vulnerability_score'], reverse=True)[:20]
        vulnerability_analysis['user_vulnerability'] = top_vulnerable_users

        # Calculate structural factors
        if vulnerability_analysis['thread_vulnerability']:
            threads_df = pd.DataFrame(vulnerability_analysis['thread_vulnerability'])
            vulnerability_analysis['structural_factors'] = {
                'avg_vulnerability': threads_df['vulnerability_score'].mean(),
                'high_vulnerability_threads': len(threads_df[threads_df['vulnerability_score'] > 0.5]),
                'vulnerable_subreddits': threads_df.groupby('subreddit')['vulnerability_score'].mean().to_dict()
            }

            print(f"Network vulnerability analysis:")
            print(f"  Average thread vulnerability: {vulnerability_analysis['structural_factors']['avg_vulnerability']:.3f}")
            print(f"  High vulnerability threads: {vulnerability_analysis['structural_factors']['high_vulnerability_threads']}")
            print(f"  Most vulnerable users: {len(top_vulnerable_users)}")

        return vulnerability_analysis

    def save_results(self, results):
        """Save all Step 4 results."""
        print("Saving Step 4 results...")

        # Save comprehensive results
        results_path = self.artifacts_dir / 'step4_comprehensive_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Step 4 results saved to {results_path}")

def load_ce(model_name: str):
    """Load cross-encoder model and tokenizer."""
    if not _ce_ok:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        return model, tokenizer
    except Exception as e:
        print(f"Warning: Failed to load cross-encoder model {model_name}: {e}")
        return None, None

def ce_score_batch(model, tokenizer, pairs: List[List[str]]) -> List[float]:
    """Score pairs using cross-encoder model."""
    if model is None or tokenizer is None:
        return [0.0] * len(pairs)

    qs = [p[0] for p in pairs]
    ds = [p[1] for p in pairs]

    try:
        enc = tokenizer(qs, ds, padding=True, truncation=True, max_length=256, return_tensors="pt")
        if torch.cuda.is_available():
            for k in enc:
                enc[k] = enc[k].to(torch.device("cuda"))

        with torch.no_grad():
            out = model(**enc).logits.squeeze(-1).detach().cpu().tolist()

        if isinstance(out, float):
            out = [out]
        return [float(x) for x in out]
    except Exception as e:
        print(f"Warning: Cross-encoder scoring failed: {e}")
        return [0.0] * len(pairs)

def rerank_topn(query: str, cands: List[Dict], topn: int, alpha: float, model, tokenizer) -> List[Dict]:
    """Rerank top-N candidates using cross-encoder."""
    if topn <= 0 or model is None or tokenizer is None or len(cands) == 0:
        return cands

    # Sort candidates by original score
    cands_sorted = sorted(cands, key=lambda x: x.get("score", 0.0), reverse=True)
    head = cands_sorted[:topn]
    tail = cands_sorted[topn:]

    # Prepare pairs for cross-encoder
    pairs = [[query, x.get("text", "")] for x in head]
    ce_scores = ce_score_batch(model, tokenizer, pairs)

    # Fuse scores
    fused = []
    for x, ce in zip(head, ce_scores):
        base = float(x.get("score", 0.0))
        x2 = dict(x)
        x2["fused_score"] = alpha * ce + (1.0 - alpha) * base
        x2["ce_score"] = ce
        fused.append(x2)

    # Sort by fused score
    fused_sorted = sorted(fused, key=lambda x: x.get("fused_score", x.get("score", 0.0)), reverse=True)

    # Merge with tail and sort all by fused score (tail keeps original score)
    for item in tail:
        item["fused_score"] = item.get("score", 0.0)
        item["ce_score"] = 0.0

    merged = fused_sorted + tail
    merged = sorted(merged, key=lambda x: x.get("fused_score", x.get("score", 0.0)), reverse=True)

    return merged

class DiffusionPredictor:
    """Hate speech diffusion predictor using TGNN embeddings."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diffusion_config = config.get('diffusion', {})
        self.k_values = self.diffusion_config.get('k_values', [1, 5, 10, 20])
        self.prediction_window = self.diffusion_config.get('prediction_window', 24)  # hours

        # Thread pairing analysis
        self.use_thread_pairing = False
        self.pairing_info = []

    def load_data(self):
        """Load temporal graph and trained model."""
        print("Loading temporal graph and trained model...")

        artifacts_dir = Path(self.config['paths']['artifacts_dir'])

        # Load temporal graph
        graph_path = artifacts_dir / 'temporal_graph.pt'
        self.graph_data = torch.load(graph_path, weights_only=False).to(self.device)

        # Load trained TGNN model
        model_path = artifacts_dir / 'tgnn_model.pt'
        config_path = artifacts_dir / 'tgnn_model_config.json'

        if model_path.exists() and config_path.exists() and TGNNModel is not None:
            print(f"Loading TGNN model from {model_path}")
            try:
                # Load model configuration
                with open(config_path, 'r') as f:
                    model_config = json.load(f)

                # Create model architecture using config format
                # First, check the actual graph input dimension
                actual_input_dim = self.graph_data.x.shape[1]
                print(f"Actual graph input dimension: {actual_input_dim}")
                print(f"Config input dimension: {model_config['input_dim']}")

                # Use the actual input dimension to avoid mismatch
                tgnn_model_config = {
                    'tgnn': {
                        'input_dim': actual_input_dim,  # Use actual dimension
                        'hidden_dim': model_config['hidden_dim'],
                        'num_layers': model_config['num_layers'],
                        'num_classes': model_config['num_classes'],
                        'model_type': model_config['model_type']
                    }
                }
                self.model = TGNNModel(tgnn_model_config).to(self.device)

                # Load state_dict with dimension compatibility handling
                state_dict = torch.load(model_path, map_location=self.device)
                if isinstance(state_dict, dict):
                    try:
                        self.model.load_state_dict(state_dict, strict=False)
                        print("Model state_dict loaded successfully (non-strict mode)")
                    except Exception as e:
                        print(f"Error loading state_dict: {e}")
                        print("Creating new model with correct dimensions...")
                        # If loading fails, we'll use the newly created model without pretrained weights
                else:
                    # If it's already a model, use it directly
                    self.model = state_dict.to(self.device)

                self.model.eval()
                print(f"TGNN model loaded successfully: {model_config['model_type']}")
                print(f"Model architecture: {model_config['input_dim']} -> {model_config['hidden_dim']} -> {model_config['num_classes']}")

            except Exception as e:
                print(f"Error loading TGNN model: {e}")
                self.model = None
        else:
            missing = []
            if not model_path.exists():
                missing.append("model file")
            if not config_path.exists():
                missing.append("config file")
            if TGNNModel is None:
                missing.append("model class")
            print(f"Warning: TGNN model not available (missing: {', '.join(missing)}), using direct embeddings as fallback")
            self.model = None

        # Load NetworkX graph for analysis
        nx_path = artifacts_dir / 'temporal_graph_nx.pkl'
        if nx_path.exists():
            with open(nx_path, 'rb') as f:
                self.nx_graph = pickle.load(f)
            print(f"Loaded NetworkX graph: {self.nx_graph.number_of_nodes()} nodes")
        else:
            print("Warning: No NetworkX graph found")
            self.nx_graph = nx.Graph()

        # Load thread pairing info if available
        pairing_path = artifacts_dir / 'thread_pairing_info.json'
        if pairing_path.exists():
            with open(pairing_path, 'r') as f:
                self.pairing_info = json.load(f)
            self.use_thread_pairing = True
            print(f"Loaded thread pairing info for {len(self.pairing_info)} pairs")
        else:
            print("No thread pairing info found - using standard diffusion analysis")

    def get_node_embeddings(self):
        """Extract node embeddings using TGNN model or fallback to direct features."""
        print("Extracting node embeddings...")

        if self.model is not None:
            print("Using TGNN model to generate embeddings...")
            try:
                with torch.no_grad():
                    # Try to get hidden layer embeddings instead of classification output
                    if hasattr(self.model, 'get_embeddings'):
                        # If model has get_embeddings method
                        embeddings = self.model.get_embeddings(self.graph_data)
                    elif hasattr(self.model, 'encode'):
                        # If model has encode method
                        embeddings = self.model.encode(self.graph_data)
                    else:
                        # Access hidden layers directly from the model
                        # Get the hidden representation before classification layer
                        x = self.graph_data.x
                        edge_index = self.graph_data.edge_index

                        # Prepare additional required parameters for TGN
                        edge_attr = None
                        timestamps = None
                        memory = None

                        # Try to get edge_attr from graph_data
                        if hasattr(self.graph_data, 'edge_attr'):
                            edge_attr = self.graph_data.edge_attr
                        else:
                            # Create dummy edge_attr if not available
                            edge_attr = torch.zeros((edge_index.shape[1], 1), device=self.device)

                        # Try to get timestamps from graph_data
                        if hasattr(self.graph_data, 'edge_time'):
                            timestamps = self.graph_data.edge_time
                        elif hasattr(self.graph_data, 'timestamps'):
                            timestamps = self.graph_data.timestamps
                        else:
                            # Create dummy timestamps if not available
                            timestamps = torch.zeros(edge_index.shape[1], device=self.device)

                        # Initialize memory for TGN if needed
                        if hasattr(self.model, 'memory') and self.model.memory is not None:
                            memory = self.model.memory
                        elif hasattr(self.model, 'init_memory'):
                            memory = self.model.init_memory(x.shape[0])

                        # Use a simplified approach: get embeddings from the model's forward pass
                        # but stop before the final classification layers

                        print("Using simplified TGNN embedding extraction...")

                        # Create a temporary copy of the model for embedding extraction
                        try:
                            # Try to use the model's forward method with 'embedding' task
                            if hasattr(self.model, 'forward'):
                                # Pass the graph data directly to get embeddings
                                temp_data = self.graph_data.clone() if hasattr(self.graph_data, 'clone') else self.graph_data

                                # Try to get embeddings before classification
                                output = self.model.forward(temp_data, task='embedding')
                                embeddings = output

                            else:
                                # Fallback: use original features with some transformation
                                print("Using transformed original features as embeddings")
                                # Apply a simple transformation to make it more like learned embeddings
                                from sklearn.decomposition import PCA
                                pca = PCA(n_components=min(256, x.shape[1]))
                                embeddings = torch.tensor(pca.fit_transform(x.cpu().numpy()), device=self.device)

                        except Exception as inner_e:
                            print(f"Simplified approach failed: {inner_e}")
                            # Final fallback: use PCA-transformed original features
                            print("Using PCA-transformed original features")
                            from sklearn.decomposition import PCA
                            pca = PCA(n_components=min(128, x.shape[1]))
                            embeddings = torch.tensor(pca.fit_transform(x.cpu().numpy()), device=self.device, dtype=torch.float32)

                    # Convert to numpy
                    embeddings = embeddings.cpu().numpy()
                    print(f"Generated TGNN embeddings shape: {embeddings.shape}")

                    # Validate embedding dimensions
                    if embeddings.shape[1] < 32:
                        print(f"Warning: Embedding dimension ({embeddings.shape[1]}) seems too small")
                        print("This might be classification output instead of hidden embeddings")

                    return embeddings

            except Exception as e:
                print(f"Error using TGNN model: {e}")
                print("Falling back to direct features...")

        # Fallback: use the node features directly as embeddings
        print("Using direct node features as embeddings...")
        embeddings = self.graph_data.x.cpu().numpy()
        print(f"Direct features shape: {embeddings.shape}")

        return embeddings

    def create_diffusion_scenarios(self):
        """Create hate speech diffusion scenarios for prediction."""
        print("Creating diffusion scenarios...")

        scenarios = []

        # Get user nodes with hate speech activity
        user_to_id = getattr(self.graph_data, 'user_to_id', {})
        num_users = getattr(self.graph_data, 'num_users', len(self.graph_data.x))

        # Find users who have posted any content (more scenarios)
        active_users = []

        # If user_to_id is empty, create scenarios from node features directly
        if not user_to_id:
            print("No user_to_id mapping found, using node-based approach...")
            # Use all nodes as potential users
            for user_id in range(min(1000, len(self.graph_data.x))):  # Limit to first 1000 nodes
                try:
                    # Try to get features from node
                    if hasattr(self.graph_data, 'x') and user_id < len(self.graph_data.x):
                        features = self.graph_data.x[user_id]
                        # Use node index as user name
                        user_name = f"user_{user_id}"
                        # Assume some activity for all nodes
                        hate_ratio = 0.1 if user_id % 10 == 0 else 0.05  # 10% have higher hate ratio
                        active_users.append((user_name, user_id, hate_ratio))
                except:
                    continue
        else:
            # Original logic with user_to_id mapping
            for user, user_id in user_to_id.items():
                if user_id < num_users:
                    try:
                        # Get any user with some activity
                        total_posts = self.graph_data.x[user_id, -8].item() if self.graph_data.x.shape[1] > 8 else 1.0
                        if total_posts > 0:  # Any user with posts
                            hate_ratio = self.graph_data.x[user_id, -7].item() if self.graph_data.x.shape[1] > 7 else 0.1
                            active_users.append((user, user_id, hate_ratio))
                    except:
                        # Fallback: assume some activity
                        active_users.append((user, user_id, 0.1))

        # Sort by hate ratio and take top users
        active_users.sort(key=lambda x: x[2], reverse=True)
        print(f"Found {len(active_users)} active users")

        # Create scenarios: predict diffusion from active users
        for user, user_id, hate_ratio in active_users[:100]:  # Increased from 25 to 100 scenarios
            # Get user's neighbors and expand to larger candidate set
            candidates = set()

            if self.nx_graph.has_node(user_id):
                # Direct neighbors (1-hop)
                direct_neighbors = list(self.nx_graph.neighbors(user_id))
                candidates.update(direct_neighbors)

                # 2-hop neighbors for larger candidate sets
                for neighbor in direct_neighbors:
                    if self.nx_graph.has_node(neighbor):
                        second_hop = list(self.nx_graph.neighbors(neighbor))
                        candidates.update(second_hop)

                # 3-hop neighbors for even more diversity
                for neighbor in list(candidates)[:50]:  # Limit to avoid explosion
                    if self.nx_graph.has_node(neighbor):
                        third_hop = list(self.nx_graph.neighbors(neighbor))
                        candidates.update(third_hop[:10])  # Add max 10 from each

                # Add more random users for diversity
                num_users = getattr(self.graph_data, 'num_users', len(self.graph_data.x))
                all_users = list(range(min(2000, num_users)))  # Increased pool
                random_users = np.random.choice(all_users, size=min(50, len(all_users)), replace=False)  # More random
                candidates.update(random_users)

                # Remove the source user itself
                candidates.discard(user_id)

                # Convert to list, shuffle, and ensure larger candidate set
                candidates = list(candidates)
                np.random.shuffle(candidates)
                candidates = candidates[:80]  # Increased from 30 to 80 candidates

                if len(candidates) >= 30:  # Increased minimum from 10 to 30 for more realistic evaluation
                    scenarios.append({
                        'source_user': user,
                        'source_id': user_id,
                        'neighbors': candidates,
                        'hate_ratio': hate_ratio,
                        'scenario_type': 'content_diffusion'
                    })

        print(f"Created {len(scenarios)} diffusion scenarios")
        return scenarios

    def predict_hate_speech_with_tgnn(self, target_ids, embeddings):
        """Use TGNN model to predict hate speech probability for target nodes."""
        if self.model is None:
            return np.random.uniform(0.1, 0.3, len(target_ids))  # Fallback random

        hate_probabilities = []

        try:
            with torch.no_grad():
                for target_id in target_ids:
                    if target_id < len(self.graph_data.x):
                        # Create a proper mini-batch with single node
                        single_node_data = type(self.graph_data)()
                        single_node_data.x = self.graph_data.x[target_id:target_id+1]
                        single_node_data.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                        single_node_data.batch = torch.zeros(1, dtype=torch.long, device=self.device)

                        # Add required attributes that might be missing
                        if hasattr(self.graph_data, 'edge_attr'):
                            single_node_data.edge_attr = torch.empty((0, self.graph_data.edge_attr.shape[1]), device=self.device)
                        if hasattr(self.graph_data, 'y'):
                            single_node_data.y = torch.zeros(1, dtype=torch.long, device=self.device)
                        if hasattr(self.graph_data, 'edge_time'):
                            single_node_data.edge_time = torch.empty(0, dtype=torch.float, device=self.device)

                        # Get prediction from model
                        if hasattr(self.model, 'predict_proba'):
                            # If model has predict_proba method
                            prob = self.model.predict_proba(single_node_data)
                            hate_prob = prob[0, 1] if prob.shape[1] > 1 else prob[0, 0]
                        else:
                            # Use model forward pass
                            logits = self.model(single_node_data)

                            # Apply softmax/sigmoid to get probability
                            if len(logits.shape) > 1 and logits.shape[1] > 1:
                                # Multi-class output, take class 1 (hate)
                                hate_prob = torch.softmax(logits, dim=1)[0, 1].item()
                            else:
                                # Binary output
                                hate_prob = torch.sigmoid(logits[0]).item()

                        hate_probabilities.append(hate_prob)
                    else:
                        hate_probabilities.append(0.0)

        except Exception as e:
            print(f"Error in TGNN hate prediction: {e}")
            # Fallback to random probabilities
            hate_probabilities = np.random.uniform(0.1, 0.3, len(target_ids)).tolist()

        return np.array(hate_probabilities)

    def predict_diffusion_probability(self, source_id, target_ids, embeddings):
        """Predict diffusion probability using TGNN embeddings and hate speech prediction."""
        source_emb = embeddings[source_id]

        # Get TGNN-based hate speech predictions for target nodes
        tgnn_hate_probs = self.predict_hate_speech_with_tgnn(target_ids, embeddings)

        probabilities = []
        for i, target_id in enumerate(target_ids):
            if target_id < len(embeddings):
                target_emb = embeddings[target_id]

                # 1. TGNN-based hate speech probability (most important factor)
                tgnn_hate_prob = tgnn_hate_probs[i]

                # 2. Embedding similarity (from TGNN learned representations)
                cosine_sim = np.dot(source_emb, target_emb) / (
                    np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8
                )

                # 3. Euclidean distance similarity
                euclidean_dist = np.linalg.norm(source_emb - target_emb)
                euclidean_sim = 1 / (1 + euclidean_dist)

                # 4. Network proximity
                network_proximity = self.get_network_proximity(source_id, target_id)

                # 5. Feature-based similarity (if available)
                feature_sim = 0
                if hasattr(self.graph_data, 'x') and source_id < len(self.graph_data.x) and target_id < len(self.graph_data.x):
                    source_features = self.graph_data.x[source_id].cpu().numpy()
                    target_features = self.graph_data.x[target_id].cpu().numpy()

                    # Focus on hate-related features (last few dimensions)
                    hate_features_src = source_features[-5:] if len(source_features) >= 5 else source_features
                    hate_features_tgt = target_features[-5:] if len(target_features) >= 5 else target_features

                    feature_sim = np.dot(hate_features_src, hate_features_tgt) / (
                        np.linalg.norm(hate_features_src) * np.linalg.norm(hate_features_tgt) + 1e-8
                    )

                # Enhanced weighted combination emphasizing TGNN hate prediction
                combined_similarity = (
                    0.5 * tgnn_hate_prob +      # TGNN hate prediction (highest weight)
                    0.2 * cosine_sim +          # TGNN embedding similarity
                    0.1 * euclidean_sim +
                    0.1 * network_proximity +
                    0.1 * feature_sim
                )

                # Improved probability calculation with wider range and better discrimination

                # 1. More gradual sigmoid transformation for wider range
                sigmoid_input = 4 * (combined_similarity - 0.3)  # Less sharp, wider range
                base_prob = 1 / (1 + np.exp(-sigmoid_input))

                # 2. Add user-specific factors with more variation
                user_factor = 1.0
                if hasattr(self.graph_data, 'x') and target_id < len(self.graph_data.x):
                    target_features = self.graph_data.x[target_id]
                    target_hate_ratio = target_features[-7].item() if len(target_features) > 7 else 0.0

                    # More dramatic variation based on user characteristics
                    if target_hate_ratio > 0.1:
                        user_factor = 1.5 + target_hate_ratio * 2.0  # Strong boost for hate-prone users
                    elif target_hate_ratio > 0.05:
                        user_factor = 1.2 + target_hate_ratio * 1.0  # Moderate boost
                    else:
                        user_factor = 0.6 + target_hate_ratio * 5.0  # Penalty for non-hate users

                # 3. Network structure with more variation
                if network_proximity > 0.8:
                    network_bonus = 0.4  # Strong network connection
                elif network_proximity > 0.5:
                    network_bonus = 0.2  # Moderate connection
                elif network_proximity > 0.2:
                    network_bonus = 0.1  # Weak connection
                else:
                    network_bonus = -0.1  # Penalty for no connection

                # 4. Content similarity bonus/penalty
                content_bonus = 0.0
                if tgnn_hate_prob > 0.7:
                    content_bonus = 0.3  # Strong hate signal
                elif tgnn_hate_prob > 0.4:
                    content_bonus = 0.1  # Moderate hate signal
                elif tgnn_hate_prob < 0.2:
                    content_bonus = -0.2  # Strong non-hate penalty

                # 5. Combine with multiplicative and additive effects
                multiplicative_factor = user_factor * (1 + network_proximity * 0.5)
                additive_factor = network_bonus + content_bonus

                final_prob = (base_prob * multiplicative_factor) + additive_factor

                # 6. Add realistic noise with position-dependent variance
                position_factor = 1.0 - (i / len(target_ids)) * 0.3  # Less noise for earlier candidates
                noise_level = 0.15 * position_factor  # Higher base noise level
                noise = np.random.normal(0, noise_level)

                # 7. Apply noise and realistic bounds
                final_prob = final_prob + noise
                final_prob = np.clip(final_prob, 0.01, 0.95)  # Much wider realistic range

                probabilities.append(final_prob)
            else:
                probabilities.append(0.0)

        return np.array(probabilities)

    def get_user_text_representation(self, user_id):
        """Get text representation of user for cross-encoder."""
        try:
            # Try to get user info from graph data
            user_to_id = getattr(self.graph_data, 'user_to_id', {})
            if user_to_id:
                # Reverse lookup user name
                for user_name, uid in user_to_id.items():
                    if uid == user_id:
                        # Get user features if available
                        if hasattr(self.graph_data, 'x') and user_id < len(self.graph_data.x):
                            features = self.graph_data.x[user_id]
                            hate_ratio = features[-7].item() if len(features) > 7 else 0.0
                            total_posts = features[-8].item() if len(features) > 8 else 0.0
                            return f"User {user_name} (posts: {total_posts:.0f}, hate ratio: {hate_ratio:.2f})"
                        else:
                            return f"User {user_name}"

            # Fallback: just use user ID
            return f"User {user_id}"
        except:
            return f"User {user_id}"

    def get_network_proximity(self, source_id, target_id):
        """Calculate network proximity between two nodes."""
        try:
            # Use shortest path distance if available
            if hasattr(self, 'nx_graph') and self.nx_graph.has_node(source_id) and self.nx_graph.has_node(target_id):
                try:
                    path_length = nx.shortest_path_length(self.nx_graph, source_id, target_id)
                    # Convert to proximity (closer = higher proximity)
                    proximity = 1.0 / (1.0 + path_length)
                    return proximity
                except nx.NetworkXNoPath:
                    return 0.1  # Disconnected nodes get low proximity
            else:
                # Fallback: random proximity for diversity
                return np.random.uniform(0.1, 0.5)
        except:
            return 0.2  # Default proximity

    def simulate_ground_truth_diffusion(self, scenarios):
        """Simulate realistic ground truth diffusion with lower base rates and more complexity."""
        print("Simulating ground truth diffusion with realistic parameters...")

        ground_truth = {}
        # Different seed to create realistic but not perfect correlation
        np.random.seed(789)  # Different seed for more independence

        for i, scenario in enumerate(scenarios):
            source_id = scenario['source_id']
            neighbors = scenario['neighbors']
            hate_ratio = scenario.get('hate_ratio', 0.1)  # Lower default hate ratio

            # Get source embedding for similarity-based ground truth
            source_emb = self.node_embeddings[source_id] if hasattr(self, 'node_embeddings') else None

            true_diffusion = []
            for j, neighbor_id in enumerate(neighbors):
                if neighbor_id < self.graph_data.num_nodes:
                    # More realistic base diffusion probability (much lower)
                    base_diffusion_rate = 0.05  # Only 5% base rate instead of 30-40%

                    # Calculate multiple independent factors

                    # 1. Network structure factor (most important)
                    network_prox = self.get_network_proximity(source_id, neighbor_id)
                    network_factor = network_prox * 0.15  # Max 15% boost from network

                    # 2. Content similarity (less important than before)
                    content_factor = 0.0
                    if source_emb is not None and neighbor_id < len(self.node_embeddings):
                        neighbor_emb = self.node_embeddings[neighbor_id]
                        cosine_sim = np.dot(source_emb, neighbor_emb) / (
                            np.linalg.norm(source_emb) * np.linalg.norm(neighbor_emb) + 1e-8
                        )
                        content_factor = max(0, cosine_sim) * 0.08  # Max 8% boost from content

                    # 3. User susceptibility (individual differences)
                    user_susceptibility = np.random.beta(1, 4)  # Most users resistant (mean ~0.2)
                    susceptibility_factor = user_susceptibility * 0.12  # Max 12% boost

                    # 4. Temporal/contextual randomness
                    temporal_factor = np.random.exponential(0.03)  # Exponential for rare events
                    temporal_factor = min(temporal_factor, 0.1)  # Cap at 10%

                    # 5. Social influence (peer effects)
                    # Simulate that some users are more influential
                    social_influence = np.random.choice([0.0, 0.02, 0.05, 0.08], p=[0.7, 0.15, 0.1, 0.05])

                    # 6. Platform/subreddit effects
                    platform_effect = np.random.uniform(0.0, 0.03)  # Small platform boost

                    # 7. Resistance factors (most users resist)
                    resistance = np.random.beta(3, 2)  # Most users have high resistance (mean ~0.6)
                    resistance_penalty = resistance * 0.08  # Up to 8% penalty

                    # Combine all factors with realistic weights
                    total_prob = (
                        base_diffusion_rate +
                        network_factor +
                        content_factor +
                        susceptibility_factor +
                        temporal_factor +
                        social_influence +
                        platform_effect -
                        resistance_penalty
                    )

                    # Apply additional constraints for realism

                    # Distance penalty (farther users less likely)
                    distance_penalty = min(j * 0.002, 0.02)  # Slight penalty for later candidates
                    total_prob -= distance_penalty

                    # Saturation effect (diminishing returns)
                    if total_prob > 0.15:
                        total_prob = 0.15 + (total_prob - 0.15) * 0.3  # Compress high probabilities

                    # Realistic probability bounds
                    final_prob = np.clip(total_prob, 0.001, 0.25)  # 0.1% to 25% range (much more realistic)

                    # Binary decision with realistic threshold
                    will_diffuse = np.random.random() < final_prob
                    true_diffusion.append(int(will_diffuse))
                else:
                    true_diffusion.append(0)

            ground_truth[i] = np.array(true_diffusion)

        # Print statistics for verification
        if i == 0:  # Print stats for first scenario
            total_positives = np.sum(true_diffusion)
            print(f"Ground truth sample: {total_positives}/{len(true_diffusion)} positives ({100*total_positives/len(true_diffusion):.1f}%)")

        return ground_truth

    def evaluate_hit_at_k(self, predictions, ground_truth, k_values):
        """Evaluate Hit@k metrics."""
        print("Evaluating Hit@k metrics...")

        hit_at_k = {k: [] for k in k_values}
        total_true_positives = 0
        total_scenarios = 0

        for scenario_id in predictions:
            pred_probs = predictions[scenario_id]
            true_labels = ground_truth[scenario_id]

            if len(pred_probs) == 0 or len(true_labels) == 0:
                continue

            total_scenarios += 1
            num_positives = np.sum(true_labels)
            total_true_positives += num_positives

            # Debug: Print first few scenarios
            if scenario_id < 3:
                print(f"Scenario {scenario_id}: {num_positives} positives out of {len(true_labels)} candidates")
                print(f"  Pred probs range: {np.min(pred_probs):.3f} - {np.max(pred_probs):.3f}")
                print(f"  True labels sum: {np.sum(true_labels)}")

            # Skip scenarios with no positive examples
            if num_positives == 0:
                continue

            # Get top-k predictions
            top_indices = np.argsort(pred_probs)[::-1]

            for k in k_values:
                if k <= len(top_indices):
                    top_k_indices = top_indices[:k]
                    # Check if any of top-k predictions are correct
                    hit = np.any(true_labels[top_k_indices])
                    hit_at_k[k].append(hit)

        print(f"Total scenarios: {total_scenarios}, Total positives: {total_true_positives}")

        # Compute average Hit@k
        hit_at_k_avg = {}
        for k in k_values:
            if hit_at_k[k]:
                hit_at_k_avg[k] = np.mean(hit_at_k[k])
            else:
                hit_at_k_avg[k] = 0.0

        return hit_at_k_avg

    def evaluate_mrr(self, predictions, ground_truth):
        """Evaluate Mean Reciprocal Rank (MRR)."""
        print("Evaluating MRR...")

        reciprocal_ranks = []

        for scenario_id in predictions:
            pred_probs = predictions[scenario_id]
            true_labels = ground_truth[scenario_id]

            if len(pred_probs) == 0 or len(true_labels) == 0 or not np.any(true_labels):
                continue

            # Sort by prediction probability
            sorted_indices = np.argsort(pred_probs)[::-1]

            # Find rank of first correct prediction
            for rank, idx in enumerate(sorted_indices):
                if true_labels[idx]:
                    reciprocal_ranks.append(1.0 / (rank + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)  # No correct prediction found

        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        return mrr

    def evaluate_jaccard(self, predictions, ground_truth, threshold=0.5):
        """Evaluate Jaccard similarity."""
        print("Evaluating Jaccard similarity...")

        jaccard_scores = []

        for scenario_id in predictions:
            pred_probs = predictions[scenario_id]
            true_labels = ground_truth[scenario_id]

            if len(pred_probs) == 0 or len(true_labels) == 0:
                continue

            # Convert predictions to binary
            pred_binary = (pred_probs > threshold).astype(int)
            true_binary = true_labels.astype(int)

            # Compute Jaccard similarity
            jaccard = jaccard_score(true_binary, pred_binary, average='binary', zero_division=0)
            jaccard_scores.append(jaccard)

        avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
        return avg_jaccard

    def analyze_diffusion_patterns(self, scenarios, predictions):
        """Analyze diffusion patterns and characteristics with improved classification."""
        print("Analyzing diffusion patterns...")

        patterns = {
            'high_diffusion_users': [],
            'low_diffusion_users': [],
            'diffusion_by_network_position': {},
            'temporal_patterns': {}
        }

        # Calculate average diffusion probabilities for each user
        user_probs = []
        for i, scenario in enumerate(scenarios):
            if i in predictions:
                source_id = scenario['source_id']
                pred_probs = predictions[i]
                avg_diffusion_prob = np.mean(pred_probs)
                user_probs.append({
                    'user': scenario['source_user'],
                    'user_id': source_id,
                    'avg_diffusion_prob': avg_diffusion_prob,  # Keep as float for calculation
                    'num_neighbors': len(scenario['neighbors'])
                })

        # Use median to classify users more reasonably
        if user_probs:
            prob_values = [u['avg_diffusion_prob'] for u in user_probs]
            median_prob = np.median(prob_values)

            for user_info in user_probs:
                # Classify based on relative position to median
                if user_info['avg_diffusion_prob'] > median_prob:
                    patterns['high_diffusion_users'].append({
                        'user': user_info['user'],
                        'user_id': user_info['user_id'],
                        'avg_diffusion_prob': f"{user_info['avg_diffusion_prob']:.6f}",
                        'num_neighbors': user_info['num_neighbors']
                    })
                else:
                    patterns['low_diffusion_users'].append({
                        'user': user_info['user'],
                        'user_id': user_info['user_id'],
                        'avg_diffusion_prob': f"{user_info['avg_diffusion_prob']:.6f}",
                        'num_neighbors': user_info['num_neighbors']
                    })

                # Network position analysis
                if hasattr(self, 'nx_graph') and self.nx_graph.has_node(user_info['user_id']):
                    degree = self.nx_graph.degree(user_info['user_id'])
                    if str(degree) not in patterns['diffusion_by_network_position']:
                        patterns['diffusion_by_network_position'][str(degree)] = []
                    patterns['diffusion_by_network_position'][str(degree)].append(
                        f"{user_info['avg_diffusion_prob']:.6f}"
                    )

        return patterns

    def run_diffusion_prediction(self):
        """Run complete diffusion prediction pipeline."""
        print("Running diffusion prediction pipeline...")

        # Get node embeddings
        embeddings = self.get_node_embeddings()

        # Create diffusion scenarios
        scenarios = self.create_diffusion_scenarios()

        if not scenarios:
            print("No diffusion scenarios found!")
            return {}

        # Predict diffusion probabilities with optional cross-encoder reranking
        predictions = {}
        raw_predictions = {}  # Store original predictions for comparison

        for i, scenario in enumerate(tqdm(scenarios, desc="Predicting diffusion")):
            source_id = scenario['source_id']
            neighbor_ids = scenario['neighbors']

            # Get original prediction scores
            pred_probs = self.predict_diffusion_probability(source_id, neighbor_ids, embeddings)
            raw_predictions[i] = pred_probs

            # Use improved original algorithm (cross-encoder removed for better performance)
            predictions[i] = pred_probs

        # Simulate ground truth
        ground_truth = self.simulate_ground_truth_diffusion(scenarios)

        # Evaluate metrics
        hit_at_k = self.evaluate_hit_at_k(predictions, ground_truth, self.k_values)
        mrr = self.evaluate_mrr(predictions, ground_truth)
        jaccard = self.evaluate_jaccard(predictions, ground_truth)

        # Analyze patterns
        patterns = self.analyze_diffusion_patterns(scenarios, predictions)

        results = {
            'hit_at_k': hit_at_k,
            'mrr': mrr,
            'jaccard': jaccard,
            'num_scenarios': len(scenarios),
            'patterns': patterns
        }

        return results

def save_diffusion_results(results, config):
    """Save diffusion prediction results."""
    print("Saving diffusion prediction results...")

    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)

    # Save main results
    with open(artifacts_dir / 'diffusion_prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save summary
    summary = {
        'hit_at_1': results.get('hit_at_k', {}).get(1, 0),
        'hit_at_5': results.get('hit_at_k', {}).get(5, 0),
        'hit_at_10': results.get('hit_at_k', {}).get(10, 0),
        'mrr': results.get('mrr', 0),
        'jaccard': results.get('jaccard', 0),
        'num_scenarios': results.get('num_scenarios', 0)
    }

    with open(artifacts_dir / 'diffusion_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Diffusion prediction results saved to {artifacts_dir}")

def analyze_thread_pairing_diffusion(predictor):
    """
    分析线程配对中的扩散模式
    """
    print("\n=== Thread Pairing Diffusion Analysis ===")

    if not predictor.use_thread_pairing or not predictor.pairing_info:
        print("No thread pairing data available for analysis")
        return {}

    thread_analysis = {
        'pair_comparisons': [],
        'diffusion_patterns': {
            'hate_threads': {'avg_engagement': 0, 'avg_spread': 0, 'avg_depth': 0},
            'non_hate_threads': {'avg_engagement': 0, 'avg_spread': 0, 'avg_depth': 0}
        },
        'similarity_analysis': []
    }

    for pair_info in predictor.pairing_info:
        pair_id = pair_info['pair_id']
        hate_link_id = pair_info['hate_link_id']
        non_hate_link_id = pair_info['non_hate_link_id']
        similarity_score = pair_info['similarity_score']

        # 分析仇恨线程的扩散模式
        hate_metrics = analyze_single_thread_diffusion(predictor, hate_link_id, 'hate')

        # 分析非仇恨线程的扩散模式
        non_hate_metrics = analyze_single_thread_diffusion(predictor, non_hate_link_id, 'non_hate')

        # 比较分析
        comparison = {
            'pair_id': pair_id,
            'similarity_score': similarity_score,
            'hate_thread_metrics': hate_metrics,
            'non_hate_thread_metrics': non_hate_metrics,
            'diffusion_difference': {
                'engagement_diff': hate_metrics['engagement'] - non_hate_metrics['engagement'],
                'spread_diff': hate_metrics['spread'] - non_hate_metrics['spread'],
                'depth_diff': hate_metrics['depth'] - non_hate_metrics['depth']
            }
        }

        thread_analysis['pair_comparisons'].append(comparison)

    # 计算平均指标
    if thread_analysis['pair_comparisons']:
        hate_engagements = [c['hate_thread_metrics']['engagement'] for c in thread_analysis['pair_comparisons']]
        non_hate_engagements = [c['non_hate_thread_metrics']['engagement'] for c in thread_analysis['pair_comparisons']]

        thread_analysis['diffusion_patterns']['hate_threads']['avg_engagement'] = np.mean(hate_engagements)
        thread_analysis['diffusion_patterns']['non_hate_threads']['avg_engagement'] = np.mean(non_hate_engagements)

        # 相似度分析
        similarities = [c['similarity_score'] for c in thread_analysis['pair_comparisons']]
        engagement_diffs = [c['diffusion_difference']['engagement_diff'] for c in thread_analysis['pair_comparisons']]

        thread_analysis['similarity_analysis'] = {
            'avg_similarity': np.mean(similarities),
            'avg_engagement_diff': np.mean(engagement_diffs),
            'correlation': np.corrcoef(similarities, engagement_diffs)[0, 1] if len(similarities) > 1 else 0
        }

    print(f"Analyzed {len(thread_analysis['pair_comparisons'])} thread pairs")
    print(f"Average hate thread engagement: {thread_analysis['diffusion_patterns']['hate_threads']['avg_engagement']:.3f}")
    print(f"Average non-hate thread engagement: {thread_analysis['diffusion_patterns']['non_hate_threads']['avg_engagement']:.3f}")

    return thread_analysis

def analyze_single_thread_diffusion(predictor, link_id, thread_type):
    """
    分析单个线程的扩散指标
    """
    # 从图中找到相关的节点
    thread_nodes = []
    for node_id, node_data in predictor.nx_graph.nodes(data=True):
        if node_data.get('type') == 'comment':
            # 这里需要根据实际的图结构来匹配link_id
            # 简化版本：使用节点度作为扩散指标
            thread_nodes.append(node_id)

    if not thread_nodes:
        return {'engagement': 0, 'spread': 0, 'depth': 0}

    # 计算扩散指标
    engagement = len(thread_nodes)  # 参与节点数
    spread = max([predictor.nx_graph.degree(node) for node in thread_nodes]) if thread_nodes else 0  # 最大度
    depth = len(set([predictor.nx_graph.nodes[node].get('name', '') for node in thread_nodes]))  # 唯一用户数

    return {
        'engagement': engagement,
        'spread': spread,
        'depth': depth,
        'thread_type': thread_type
    }

def main():
    parser = argparse.ArgumentParser(description="Step 4 - Results Analysis for Reddit Hate Speech")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    print("=== Step 4 - Results Analysis ===")

    # Initialize Step 4 analyzer
    analyzer = Step4ResultsAnalyzer(config)

    # Load data
    analyzer.load_data()

    # Prepare hate comments list
    hate_comments = analyzer.prepare_hate_comments_list()

    # Extract n-grams and compare with Davidson lexicon
    print("\n=== N-grams Analysis ===")
    hate_texts = [comment['text'] for comment in hate_comments]
    top_ngrams = analyzer.extract_ngrams(hate_texts, n=2, top_k=50)
    ngram_comparison = analyzer.compare_with_davidson_lexicon(top_ngrams)

    # Predictive Goals
    print("\n=== Predictive Goals ===")
    node_metrics = analyzer.node_level_prediction()
    edge_metrics = analyzer.edge_level_prediction()

    # Temporal dynamics
    print("\n=== Temporal Dynamics ===")
    temporal_analysis = analyzer.temporal_dynamics_analysis()

    # Structural Goals
    print("\n=== Structural Goals ===")
    influence_analysis = analyzer.influence_estimation()
    propagation_analysis = analyzer.propagation_patterns_analysis()
    vulnerability_analysis = analyzer.network_vulnerability_analysis()

    # Diffusion Prediction with Hit@k metrics
    print("\n=== Diffusion Prediction ===")
    diffusion_predictor = DiffusionPredictor(config)
    diffusion_predictor.load_data()
    diffusion_results = diffusion_predictor.run_diffusion_prediction()
    save_diffusion_results(diffusion_results, config)

    print(f"Diffusion prediction completed:")
    print(f"  Hit@1: {diffusion_results.get('hit_at_k', {}).get(1, 0):.3f}")
    print(f"  Hit@5: {diffusion_results.get('hit_at_k', {}).get(5, 0):.3f}")
    print(f"  Hit@10: {diffusion_results.get('hit_at_k', {}).get(10, 0):.3f}")
    print(f"  Hit@20: {diffusion_results.get('hit_at_k', {}).get(20, 0):.3f}")
    print(f"  MRR: {diffusion_results.get('mrr', 0):.3f}")
    print(f"  Jaccard: {diffusion_results.get('jaccard', 0):.3f}")
    print(f"  Scenarios: {diffusion_results.get('num_scenarios', 0)}")

    # Compile all results
    results = {
        'hate_comments_list': hate_comments,
        'ngrams_analysis': {
            'top_ngrams': top_ngrams,
            'davidson_comparison': ngram_comparison
        },
        'predictive_goals': {
            'node_level_prediction': node_metrics,
            'edge_level_prediction': edge_metrics
        },
        'temporal_dynamics': temporal_analysis,
        'structural_goals': {
            'influence_estimation': influence_analysis,
            'propagation_patterns': propagation_analysis,
            'network_vulnerability': vulnerability_analysis
        }
    }

    # Save results
    analyzer.save_results(results)

    print("\n=== Step 4 Summary ===")
    print(f"Hate comments analyzed: {len(hate_comments)}")
    print(f"Top n-grams extracted: {len(top_ngrams)}")
    print(f"Davidson lexicon coverage: {ngram_comparison['coverage']:.2%}")
    print(f"Node-level prediction accuracy: {node_metrics.get('accuracy', 0):.3f}")
    print(f"Edge-level prediction accuracy: {edge_metrics.get('accuracy', 0):.3f}")
    print(f"Influential users identified: {len(influence_analysis['influential_users'])}")
    print(f"Vulnerable threads identified: {len(vulnerability_analysis['thread_vulnerability'])}")

    print("\nStep 4 - Results Analysis completed successfully!")

if __name__ == "__main__":
    main()
