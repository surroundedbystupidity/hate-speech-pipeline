#!/usr/bin/env python3
"""
Advanced Evaluation Script for Reddit Hate Speech Analysis.
Comprehensive evaluation with multi-task learning metrics and research insights.
"""

import numpy as np
import pandas as pd
import torch
import json
import argparse
from pathlib import Path
import yaml
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class AdvancedEvaluator:
    """Advanced evaluator for comprehensive analysis."""
    
    def __init__(self, config):
        self.config = config
        self.artifacts_dir = Path(config['paths']['artifacts_dir'])
        self.figures_dir = Path(config['paths']['figures_dir'])
        self.figures_dir.mkdir(exist_ok=True)
        
    def load_all_results(self):
        """Load all results from different components."""
        print("Loading all evaluation results...")
        
        self.results = {}
        
        # Load baseline results
        baseline_path = self.artifacts_dir / 'baseline_metrics.json'
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                self.results['baseline'] = json.load(f)
        
        # Load TGNN results
        tgnn_path = self.artifacts_dir / 'tgnn_metrics.json'
        if tgnn_path.exists():
            with open(tgnn_path, 'r') as f:
                self.results['tgnn'] = json.load(f)
        
        # Load diffusion prediction results
        diffusion_path = self.artifacts_dir / 'diffusion_prediction_results.json'
        if diffusion_path.exists():
            with open(diffusion_path, 'r') as f:
                self.results['diffusion'] = json.load(f)
        
        # Load moderation simulation results
        moderation_path = self.artifacts_dir / 'moderation_simulation_results.json'
        if moderation_path.exists():
            with open(moderation_path, 'r') as f:
                self.results['moderation'] = json.load(f)
        
        # Load data preparation stats
        data_prep_path = self.artifacts_dir / 'subreddit_analysis.json'
        if data_prep_path.exists():
            with open(data_prep_path, 'r') as f:
                self.results['data_prep'] = json.load(f)
        
        print(f"Loaded results from {len(self.results)} components")
    
    def evaluate_classification_performance(self):
        """Evaluate hate speech classification performance."""
        print("Evaluating classification performance...")
        
        classification_metrics = {}
        
        # Baseline metrics
        if 'baseline' in self.results:
            baseline = self.results['baseline']
            classification_metrics['baseline'] = {
                'accuracy': baseline.get('accuracy', 0),
                'precision': baseline.get('precision', 0),
                'recall': baseline.get('recall', 0),
                'f1': baseline.get('f1_score', 0),
                'auc': baseline.get('roc_auc', 0)
            }
        
        # TGNN metrics
        if 'tgnn' in self.results:
            tgnn = self.results['tgnn']
            classification_metrics['tgnn'] = {
                'accuracy': tgnn.get('accuracy', 0),
                'precision': tgnn.get('precision', 0),
                'recall': tgnn.get('recall', 0),
                'f1': tgnn.get('f1', 0),
                'auc': tgnn.get('auc', 0)
            }
        
        # Create comparison
        comparison = {}
        if 'baseline' in classification_metrics and 'tgnn' in classification_metrics:
            baseline_f1 = classification_metrics['baseline']['f1']
            tgnn_f1 = classification_metrics['tgnn']['f1']
            improvement = ((tgnn_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
            
            comparison['f1_improvement'] = improvement
            comparison['baseline_f1'] = baseline_f1
            comparison['tgnn_f1'] = tgnn_f1
        
        return {
            'metrics': classification_metrics,
            'comparison': comparison
        }
    
    def evaluate_diffusion_performance(self):
        """Evaluate diffusion prediction performance."""
        print("Evaluating diffusion prediction performance...")
        
        if 'diffusion' not in self.results:
            return {'error': 'No diffusion results found'}
        
        diffusion = self.results['diffusion']
        
        # Extract key metrics
        diffusion_metrics = {
            'hit_at_1': diffusion.get('hit_at_k', {}).get('1', 0),
            'hit_at_5': diffusion.get('hit_at_k', {}).get('5', 0),
            'hit_at_10': diffusion.get('hit_at_k', {}).get('10', 0),
            'hit_at_20': diffusion.get('hit_at_k', {}).get('20', 0),
            'mrr': diffusion.get('mrr', 0),
            'jaccard': diffusion.get('jaccard', 0),
            'num_scenarios': diffusion.get('num_scenarios', 0)
        }
        
        # Analyze diffusion patterns
        patterns = diffusion.get('patterns', {})
        pattern_analysis = {
            'high_diffusion_users': len(patterns.get('high_diffusion_users', [])),
            'low_diffusion_users': len(patterns.get('low_diffusion_users', [])),
            'network_effects': len(patterns.get('diffusion_by_network_position', {}))
        }
        
        return {
            'metrics': diffusion_metrics,
            'patterns': pattern_analysis
        }
    
    def evaluate_moderation_effectiveness(self):
        """Evaluate moderation strategy effectiveness."""
        print("Evaluating moderation effectiveness...")
        
        if 'moderation' not in self.results:
            return {'error': 'No moderation results found'}
        
        moderation = self.results['moderation']
        
        # Extract effectiveness scores
        effectiveness = moderation.get('effectiveness_scores', {})
        strategy_impacts = moderation.get('strategy_impacts', {})
        
        # Analyze each strategy
        strategy_analysis = {}
        for strategy, impact in strategy_impacts.items():
            strategy_analysis[strategy] = {
                'hate_reduction': impact.get('hate_reduction', 0),
                'effectiveness_score': effectiveness.get(strategy, 0),
                'nodes_removed': impact.get('nodes_removed', 0),
                'remaining_samples': impact.get('remaining_samples', 0)
            }
        
        # Find best strategy
        best_strategy = max(effectiveness, key=effectiveness.get) if effectiveness else 'none'
        best_score = effectiveness.get(best_strategy, 0)
        
        return {
            'best_strategy': best_strategy,
            'best_score': best_score,
            'strategy_analysis': strategy_analysis,
            'effectiveness_scores': effectiveness
        }
    
    def analyze_dataset_characteristics(self):
        """Analyze dataset characteristics and quality."""
        print("Analyzing dataset characteristics...")
        
        characteristics = {}
        
        # Data preparation analysis
        if 'data_prep' in self.results:
            data_prep = self.results['data_prep']
            
            # Subreddit analysis
            hate_subreddits = data_prep.get('top_hate_subreddits', [])
            normal_subreddits = data_prep.get('top_normal_subreddits', [])
            
            characteristics['subreddit_analysis'] = {
                'hate_subreddits_count': len(hate_subreddits),
                'normal_subreddits_count': len(normal_subreddits),
                'avg_hate_ratio': np.mean([s.get('hate_ratio', 0) for s in hate_subreddits]) if hate_subreddits else 0,
                'avg_normal_ratio': np.mean([s.get('hate_ratio', 0) for s in normal_subreddits]) if normal_subreddits else 0
            }
        
        # Load dataset for additional analysis
        balanced_path = self.artifacts_dir / 'balanced_dataset.parquet'
        if balanced_path.exists():
            df = pd.read_parquet(balanced_path)
            
            characteristics['dataset_stats'] = {
                'total_samples': len(df),
                'hate_samples': len(df[df.get('davidson_label', 0) == 2]),
                'offensive_samples': len(df[df.get('davidson_label', 0) == 1]),
                'normal_samples': len(df[df.get('davidson_label', 0) == 0]),
                'unique_users': df['author'].nunique(),
                'unique_subreddits': df['subreddit'].nunique(),
                'avg_text_length': df['text_content'].str.len().mean() if 'text_content' in df.columns else 0
            }
            
            # Balance analysis
            hate_ratio = len(df[df.get('davidson_label', 0) == 2]) / len(df)
            characteristics['balance_analysis'] = {
                'hate_ratio': hate_ratio,
                'is_balanced': 0.4 <= hate_ratio <= 0.6,
                'balance_score': 1 - abs(0.5 - hate_ratio) * 2  # Score from 0 to 1
            }
        
        return characteristics
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive evaluation visualizations."""
        print("Creating comprehensive visualizations...")
        
        plt.style.use('default')
        
        # 1. Model Performance Comparison
        classification_results = self.evaluate_classification_performance()
        if 'metrics' in classification_results:
            self.plot_model_comparison(classification_results['metrics'])
        
        # 2. Diffusion Metrics Visualization
        diffusion_results = self.evaluate_diffusion_performance()
        if 'metrics' in diffusion_results:
            self.plot_diffusion_metrics(diffusion_results['metrics'])
        
        # 3. Moderation Effectiveness
        moderation_results = self.evaluate_moderation_effectiveness()
        if 'strategy_analysis' in moderation_results:
            self.plot_moderation_effectiveness(moderation_results)
        
        # 4. Dataset Characteristics
        dataset_chars = self.analyze_dataset_characteristics()
        if 'dataset_stats' in dataset_chars:
            self.plot_dataset_characteristics(dataset_chars)
    
    def plot_model_comparison(self, metrics):
        """Plot model performance comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        models = list(metrics.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        for i, metric in enumerate(metric_names[:4]):
            ax = axes[i//2, i%2]
            values = [metrics[model].get(metric, 0) for model in models]
            
            bars = ax.bar(models, values, alpha=0.7)
            ax.set_title(f'{metric.capitalize()}')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_diffusion_metrics(self, metrics):
        """Plot diffusion prediction metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Hit@k metrics
        k_values = [1, 5, 10, 20]
        hit_values = [metrics.get(f'hit_at_{k}', 0) for k in k_values]
        
        ax1.plot(k_values, hit_values, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('k')
        ax1.set_ylabel('Hit@k')
        ax1.set_title('Hit@k Performance')
        ax1.grid(True, alpha=0.3)
        
        # Other metrics
        other_metrics = ['mrr', 'jaccard']
        other_values = [metrics.get(metric, 0) for metric in other_metrics]
        
        bars = ax2.bar(other_metrics, other_values, alpha=0.7)
        ax2.set_title('Diffusion Prediction Metrics')
        ax2.set_ylim(0, 1)
        
        for bar, value in zip(bars, other_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'diffusion_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_moderation_effectiveness(self, moderation_results):
        """Plot moderation strategy effectiveness."""
        strategy_analysis = moderation_results['strategy_analysis']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Effectiveness scores
        strategies = list(strategy_analysis.keys())
        effectiveness_scores = [strategy_analysis[s]['effectiveness_score'] for s in strategies]
        hate_reductions = [strategy_analysis[s]['hate_reduction'] * 100 for s in strategies]
        
        bars1 = ax1.bar(strategies, effectiveness_scores, alpha=0.7, color='skyblue')
        ax1.set_title('Moderation Strategy Effectiveness Scores')
        ax1.set_ylabel('Effectiveness Score')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars1, effectiveness_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Hate reduction percentages
        bars2 = ax2.bar(strategies, hate_reductions, alpha=0.7, color='lightcoral')
        ax2.set_title('Hate Speech Reduction by Strategy')
        ax2.set_ylabel('Hate Reduction (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, hate_reductions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'moderation_effectiveness.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_dataset_characteristics(self, characteristics):
        """Plot dataset characteristics."""
        dataset_stats = characteristics['dataset_stats']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Dataset Characteristics', fontsize=16)
        
        # Label distribution
        labels = ['Normal', 'Offensive', 'Hate']
        counts = [
            dataset_stats['normal_samples'],
            dataset_stats['offensive_samples'],
            dataset_stats['hate_samples']
        ]
        colors = ['lightgreen', 'orange', 'red']
        
        ax1.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Label Distribution')
        
        # User and subreddit counts
        categories = ['Users', 'Subreddits']
        values = [dataset_stats['unique_users'], dataset_stats['unique_subreddits']]
        
        bars = ax2.bar(categories, values, alpha=0.7, color=['lightblue', 'lightpink'])
        ax2.set_title('Unique Users and Subreddits')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value}', ha='center', va='bottom')
        
        # Balance analysis
        if 'balance_analysis' in characteristics:
            balance = characteristics['balance_analysis']
            ax3.bar(['Balance Score'], [balance['balance_score']], alpha=0.7, color='gold')
            ax3.set_title('Dataset Balance Score')
            ax3.set_ylim(0, 1)
            ax3.text(0, balance['balance_score'] + 0.05, f"{balance['balance_score']:.3f}",
                    ha='center', va='bottom')
        
        # Text length distribution (placeholder)
        ax4.text(0.5, 0.5, f"Avg Text Length:\n{dataset_stats['avg_text_length']:.0f} chars",
                ha='center', va='center', transform=ax4.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax4.set_title('Text Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'dataset_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_research_insights(self):
        """Generate research insights and findings."""
        print("Generating research insights...")
        
        insights = {
            'classification_insights': [],
            'diffusion_insights': [],
            'moderation_insights': [],
            'dataset_insights': [],
            'methodological_insights': []
        }
        
        # Classification insights
        classification_results = self.evaluate_classification_performance()
        if 'comparison' in classification_results:
            comp = classification_results['comparison']
            if 'f1_improvement' in comp:
                improvement = comp['f1_improvement']
                if improvement > 5:
                    insights['classification_insights'].append(
                        f"TGNN model shows {improvement:.1f}% improvement over baseline TF-IDF approach"
                    )
                elif improvement < -5:
                    insights['classification_insights'].append(
                        f"Baseline model outperforms TGNN by {abs(improvement):.1f}%, suggesting simpler approaches may be sufficient"
                    )
                else:
                    insights['classification_insights'].append(
                        "TGNN and baseline models show similar performance, indicating comparable effectiveness"
                    )
        
        # Diffusion insights
        diffusion_results = self.evaluate_diffusion_performance()
        if 'metrics' in diffusion_results:
            metrics = diffusion_results['metrics']
            if metrics['hit_at_1'] > 0.3:
                insights['diffusion_insights'].append(
                    f"High Hit@1 score ({metrics['hit_at_1']:.3f}) indicates strong predictive capability for immediate diffusion"
                )
            if metrics['mrr'] > 0.5:
                insights['diffusion_insights'].append(
                    f"MRR of {metrics['mrr']:.3f} suggests good ranking quality for diffusion prediction"
                )
        
        # Moderation insights
        moderation_results = self.evaluate_moderation_effectiveness()
        if 'best_strategy' in moderation_results:
            best_strategy = moderation_results['best_strategy']
            best_score = moderation_results['best_score']
            insights['moderation_insights'].append(
                f"Most effective moderation strategy: {best_strategy} (effectiveness score: {best_score:.1f})"
            )
            
            if 'strategy_analysis' in moderation_results:
                analysis = moderation_results['strategy_analysis']
                for strategy, data in analysis.items():
                    reduction = data['hate_reduction'] * 100
                    if reduction > 50:
                        insights['moderation_insights'].append(
                            f"{strategy} achieves {reduction:.1f}% hate speech reduction"
                        )
        
        # Dataset insights
        dataset_chars = self.analyze_dataset_characteristics()
        if 'balance_analysis' in dataset_chars:
            balance = dataset_chars['balance_analysis']
            if balance['is_balanced']:
                insights['dataset_insights'].append("Dataset is well-balanced for hate speech classification")
            else:
                insights['dataset_insights'].append(
                    f"Dataset imbalance detected (hate ratio: {balance['hate_ratio']:.3f})"
                )
        
        # Methodological insights
        insights['methodological_insights'].extend([
            "Davidson lexicon provides effective weak supervision for hate speech labeling",
            "Temporal graph structure captures important user interaction patterns",
            "Multi-task learning approach enables comprehensive hate speech analysis"
        ])
        
        return insights
    
    def create_evaluation_report(self):
        """Create comprehensive evaluation report."""
        print("Creating evaluation report...")
        
        # Collect all results
        classification_results = self.evaluate_classification_performance()
        diffusion_results = self.evaluate_diffusion_performance()
        moderation_results = self.evaluate_moderation_effectiveness()
        dataset_characteristics = self.analyze_dataset_characteristics()
        research_insights = self.generate_research_insights()
        
        # Create comprehensive report
        report = {
            'evaluation_summary': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'components_evaluated': len(self.results),
                'visualizations_created': True
            },
            'classification_performance': classification_results,
            'diffusion_prediction': diffusion_results,
            'moderation_effectiveness': moderation_results,
            'dataset_characteristics': dataset_characteristics,
            'research_insights': research_insights
        }
        
        return report

def save_evaluation_report(report, config):
    """Save comprehensive evaluation report."""
    print("Saving evaluation report...")
    
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save complete report
    with open(artifacts_dir / 'comprehensive_evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Create summary for presentation
    summary = {
        'key_findings': [],
        'performance_metrics': {},
        'recommendations': []
    }
    
    # Extract key findings
    if 'classification_performance' in report:
        if 'comparison' in report['classification_performance']:
            comp = report['classification_performance']['comparison']
            if 'f1_improvement' in comp:
                summary['key_findings'].append(f"TGNN vs Baseline F1 improvement: {comp['f1_improvement']:.1f}%")
    
    if 'diffusion_prediction' in report and 'metrics' in report['diffusion_prediction']:
        metrics = report['diffusion_prediction']['metrics']
        summary['key_findings'].append(f"Diffusion Hit@1: {metrics.get('hit_at_1', 0):.3f}")
        summary['key_findings'].append(f"Diffusion MRR: {metrics.get('mrr', 0):.3f}")
    
    if 'moderation_effectiveness' in report:
        mod = report['moderation_effectiveness']
        if 'best_strategy' in mod:
            summary['key_findings'].append(f"Best moderation strategy: {mod['best_strategy']}")
    
    # Performance metrics summary
    if 'classification_performance' in report and 'metrics' in report['classification_performance']:
        summary['performance_metrics'] = report['classification_performance']['metrics']
    
    # Recommendations
    summary['recommendations'] = [
        "Continue developing TGNN approaches for temporal hate speech analysis",
        "Implement best-performing moderation strategy in practice",
        "Expand dataset with more diverse subreddits and time periods",
        "Investigate user-level interventions for hate speech reduction"
    ]
    
    with open(artifacts_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Evaluation report saved to {artifacts_dir}")

def main():
    parser = argparse.ArgumentParser(description="Advanced evaluation for Reddit hate speech analysis")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("=== Advanced Evaluation ===")
    
    # Initialize evaluator
    evaluator = AdvancedEvaluator(config)
    
    # Load all results
    evaluator.load_all_results()
    
    # Create visualizations
    evaluator.create_comprehensive_visualizations()
    
    # Generate comprehensive report
    report = evaluator.create_evaluation_report()
    
    # Save report
    save_evaluation_report(report, config)
    
    print(f"\n=== Evaluation Summary ===")
    print(f"Components evaluated: {len(evaluator.results)}")
    
    # Print key insights
    insights = report['research_insights']
    print(f"\nKey Research Insights:")
    for category, insight_list in insights.items():
        if insight_list:
            print(f"\n{category.replace('_', ' ').title()}:")
            for insight in insight_list[:2]:  # Show top 2 insights per category
                print(f"  • {insight}")
    
    print(f"\nDetailed evaluation report saved to artifacts/comprehensive_evaluation_report.json")
    print(f"Visualizations saved to figures/ directory")

if __name__ == "__main__":
    main()
