#!/usr/bin/env python3
"""
Baseline Model: TF-IDF + Logistic Regression

This script implements a traditional baseline model for hate speech classification
using TF-IDF vectorization and Logistic Regression. This serves as a comparison
baseline for the more advanced TGNN models.

The baseline model provides:
- Classical text classification approach
- Fast training and inference
- Interpretable feature importance
- Standard evaluation metrics

Author: Research Team
Date: 2024
Version: 1.0
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import yaml
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, accuracy_score
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
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
        raise ValueError(f"Error parsing YAML configuration: {e}")

def load_balanced_data(config):
    """
    Load balanced dataset for baseline training.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (X, y) - texts and labels
    """
    print("Loading balanced dataset...")
    
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    data_path = artifacts_dir / 'balanced_dataset.parquet'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Balanced dataset not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    
    # Extract text content and labels
    texts = df['text_content'].fillna('').astype(str)
    labels = df['binary_label'].astype(int)
    
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution: {labels.value_counts().to_dict()}")
    
    return texts, labels

def create_baseline_pipeline(config):
    """
    Create TF-IDF + Logistic Regression pipeline.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        Pipeline: Sklearn pipeline with TF-IDF and LogisticRegression
    """
    print("Creating baseline pipeline...")
    
    # Extract baseline configuration
    baseline_config = config.get('baseline', {})
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=baseline_config.get('max_features', 10000),
        ngram_range=tuple(baseline_config.get('ngram_range', [1, 2])),
        lowercase=True,
        stop_words='english',
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    # Logistic Regression Classifier
    classifier = LogisticRegression(
        C=baseline_config.get('C', 1.0),
        class_weight=baseline_config.get('class_weight', 'balanced'),
        random_state=config.get('random_state', 42),
        max_iter=1000,
        solver='liblinear'
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('classifier', classifier)
    ])
    
    return pipeline

def train_baseline_model(pipeline, X_train, y_train, config):
    """
    Train the baseline model with hyperparameter tuning.
    
    Args:
        pipeline: Sklearn pipeline
        X_train: Training texts
        y_train: Training labels
        config: Configuration dictionary
        
    Returns:
        Pipeline: Trained pipeline
    """
    print("Training baseline model...")
    
    # Hyperparameter grid for tuning
    param_grid = {
        'tfidf__max_features': [5000, 10000, 20000],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'classifier__C': [0.1, 1.0, 10.0]
    }
    
    # Grid search with cross-validation
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_baseline_model(model, X_test, y_test):
    """
    Evaluate the baseline model performance.
    
    Args:
        model: Trained model
        X_test: Test texts
        y_test: Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    print("Evaluating baseline model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'auc_roc': auc_score,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    return metrics

def analyze_feature_importance(model, top_n=20):
    """
    Analyze and extract top features from the TF-IDF model.
    
    Args:
        model: Trained pipeline
        top_n (int): Number of top features to extract
        
    Returns:
        dict: Feature importance analysis
    """
    print(f"Analyzing top {top_n} features...")
    
    # Get TF-IDF vectorizer and classifier
    vectorizer = model.named_steps['tfidf']
    classifier = model.named_steps['classifier']
    
    # Get feature names and coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    
    # Top positive features (hate speech indicators)
    top_hate_indices = np.argsort(coefficients)[-top_n:][::-1]
    top_hate_features = [(feature_names[i], coefficients[i]) for i in top_hate_indices]
    
    # Top negative features (normal speech indicators)
    top_normal_indices = np.argsort(coefficients)[:top_n]
    top_normal_features = [(feature_names[i], coefficients[i]) for i in top_normal_indices]
    
    feature_analysis = {
        'top_hate_indicators': top_hate_features,
        'top_normal_indicators': top_normal_features,
        'total_features': len(feature_names)
    }
    
    return feature_analysis

def create_baseline_visualizations(metrics, feature_analysis, config):
    """
    Create visualizations for baseline model analysis.
    
    Args:
        metrics (dict): Evaluation metrics
        feature_analysis (dict): Feature importance analysis
        config (dict): Configuration dictionary
    """
    print("Creating baseline model visualizations...")
    
    figures_dir = Path(config['paths']['figures_dir'])
    figures_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Baseline Model - Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticklabels(['Normal', 'Hate Speech'])
    ax.set_yticklabels(['Normal', 'Hate Speech'])
    plt.tight_layout()
    plt.savefig(figures_dir / 'baseline_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top hate speech indicators
    hate_features, hate_scores = zip(*feature_analysis['top_hate_indicators'][:15])
    ax1.barh(range(len(hate_features)), hate_scores, color='red', alpha=0.7)
    ax1.set_yticks(range(len(hate_features)))
    ax1.set_yticklabels(hate_features)
    ax1.set_title('Top Hate Speech Indicators', fontweight='bold')
    ax1.set_xlabel('TF-IDF Coefficient')
    
    # Top normal speech indicators
    normal_features, normal_scores = zip(*feature_analysis['top_normal_indicators'][:15])
    ax2.barh(range(len(normal_features)), normal_scores, color='blue', alpha=0.7)
    ax2.set_yticks(range(len(normal_features)))
    ax2.set_yticklabels(normal_features)
    ax2.set_title('Top Normal Speech Indicators', fontweight='bold')
    ax2.set_xlabel('TF-IDF Coefficient')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'baseline_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance Metrics Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    metrics_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['auc_roc']
    ]
    
    bars = ax.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral'])
    ax.set_title('Baseline Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'baseline_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Baseline visualizations saved to {figures_dir}")

def save_baseline_results(model, metrics, feature_analysis, config):
    """
    Save baseline model and results.
    
    Args:
        model: Trained model
        metrics (dict): Evaluation metrics
        feature_analysis (dict): Feature importance analysis
        config (dict): Configuration dictionary
    """
    print("Saving baseline model and results...")
    
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = artifacts_dir / 'baseline_model.joblib'
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics_path = artifacts_dir / 'baseline_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Save feature analysis
    features_path = artifacts_dir / 'baseline_feature_analysis.json'
    with open(features_path, 'w') as f:
        json.dump(feature_analysis, f, indent=2, default=str)
    
    print(f"Baseline model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Feature analysis saved to {features_path}")

def main():
    """
    Main execution function for baseline model training.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + Logistic Regression baseline model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python scripts/baseline_tfidf_lr.py --config configs/exp_small.yaml
    
This script trains a traditional baseline model for comparison with TGNN models.
        """
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to YAML configuration file"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        print("=" * 60)
        print("TF-IDF + Logistic Regression Baseline Model Training")
        print("=" * 60)
        print(f"Configuration: {args.config}")
        print(f"Random seed: {config.get('random_state', 42)}")
        print("=" * 60)
        
        # Load data
        X, y = load_balanced_data(config)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=config.get('random_state', 42),
            stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Create and train model
        pipeline = create_baseline_pipeline(config)
        trained_model = train_baseline_model(pipeline, X_train, y_train, config)
        
        # Evaluate model
        metrics = evaluate_baseline_model(trained_model, X_test, y_test)
        
        # Analyze features
        feature_analysis = analyze_feature_importance(trained_model)
        
        # Create visualizations
        create_baseline_visualizations(metrics, feature_analysis, config)
        
        # Save results
        save_baseline_results(trained_model, metrics, feature_analysis, config)
        
        # Print summary
        print("\n" + "=" * 60)
        print("BASELINE MODEL TRAINING SUMMARY")
        print("=" * 60)
        print(f"Model: TF-IDF + Logistic Regression")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print("\nPerformance Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"\nTotal TF-IDF features: {feature_analysis['total_features']:,}")
        print("\nBaseline model training completed successfully!")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"Error during baseline model training: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
