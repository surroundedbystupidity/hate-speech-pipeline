# Reddit Hate Speech Diffusion Analysis

A comprehensive system for analyzing hate speech diffusion patterns on Reddit using Temporal Graph Neural Networks (TGNN). This project implements an end-to-end pipeline for hate speech detection, diffusion prediction, and moderation strategy evaluation.

## Overview

This system addresses the critical challenge of understanding how hate speech spreads through social networks. By leveraging temporal graph neural networks, we can not only detect hate speech but also predict its diffusion patterns and evaluate the effectiveness of different moderation strategies.

## Key Features

- **Hate Speech Detection**: Advanced classification using TGNN with 96.1% accuracy
- **Diffusion Prediction**: Novel algorithm to predict hate speech propagation patterns
- **Moderation Simulation**: Evaluation of different content moderation strategies
- **Comprehensive Evaluation**: Multi-dimensional metrics including Hit@k, MRR, and Jaccard similarity
- **GPU Acceleration**: Optimized for CUDA-enabled training and inference
- **Modular Design**: Clean, maintainable codebase with configurable parameters

## System Architecture

```
Data Input → Preprocessing → Feature Extraction → Graph Construction → Model Training → Diffusion Prediction → Moderation Simulation → Results
     ↓            ↓              ↓                   ↓                    ↓                ↓                     ↓                      ↓
Reddit Data   Davidson      BERT Embeddings      Temporal Graph      TGNN Model    Hit@k Evaluation        Strategy Testing        Visualizations
              Labeling
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers scikit-learn pandas numpy
pip install networkx matplotlib seaborn plotly
pip install pyyaml tqdm
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/HarlanAlternative/HateSpeechPipeline.git
```

### 2. Run the Complete Pipeline

```bash
# For GPU-accelerated processing (recommended)
run_gpu_7days.bat

# Or run individual components
python scripts/00_data_preparation.py --config configs/exp_gpu_7days.yaml
python scripts/01_bert_feature_extraction.py --config configs/exp_gpu_7days.yaml
python scripts/02_temporal_graph_construction.py --config configs/exp_gpu_7days.yaml
python scripts/03_tgnn_model.py --config configs/exp_gpu_7days.yaml
python scripts/04_diffusion_prediction.py --config configs/exp_gpu_7days.yaml
python scripts/05_moderation_simulation.py --config configs/exp_gpu_7days.yaml
python scripts/06_advanced_evaluation.py --config configs/exp_gpu_7days.yaml
python scripts/07_research_visualization.py --config configs/exp_gpu_7days.yaml
```

## Project Structure

```
reddit/
├── scripts/                          # Core implementation scripts
│   ├── 00_data_preparation.py        # Data preprocessing and labeling
│   ├── 01_bert_feature_extraction.py # BERT feature extraction
│   ├── 02_temporal_graph_construction.py # Temporal graph construction
│   ├── 03_tgnn_model.py             # TGNN model training
│   ├── 04_diffusion_prediction.py   # Diffusion prediction analysis
│   ├── 05_moderation_simulation.py  # Moderation strategy simulation
│   ├── 06_advanced_evaluation.py    # Comprehensive evaluation
│   ├── 07_research_visualization.py # Research visualizations
│   └── baseline_tfidf_lr.py         # Baseline model implementation
├── configs/                          # Configuration files
│   ├── exp_small.yaml               # Small-scale experiment config
│   └── exp_gpu_7days.yaml          # GPU-accelerated config
├── artifacts/                        # Output results and models
└── figures/                          # Generated visualizations
```

## Core Components

### 1. Data Preprocessing (`00_data_preparation.py`)

- Loads Reddit submissions and comments
- Applies Davidson lexicon for hate speech labeling
- Creates balanced 1:1 datasets
- Analyzes subreddit hate levels

### 2. Feature Extraction (`01_bert_feature_extraction.py`)

- Extracts BERT embeddings for text content
- Computes user-level features (posting frequency, hate ratio)
- Generates node embeddings for graph construction

### 3. Graph Construction (`02_temporal_graph_construction.py`)

- Creates temporal graph with users and posts as nodes
- Establishes edges based on interactions (replies, co-commenting)
- Incorporates timestamp information for temporal modeling

### 4. Model Training (`03_tgnn_model.py`)

- Implements Temporal Graph Neural Network (TGAT architecture)
- Trains on hate speech classification task
- Supports GPU acceleration for faster training

### 5. Diffusion Prediction (`04_diffusion_prediction.py`)

- Predicts hate speech propagation patterns
- Uses multi-factor similarity computation
- Evaluates using Hit@k, MRR, and Jaccard metrics

### 6. Moderation Simulation (`05_moderation_simulation.py`)

- Simulates different moderation strategies
- Evaluates effectiveness of content removal, user banning, and subreddit banning
- Provides quantitative analysis of intervention impact

## Results

### Model Performance Comparison

| Model                  | Accuracy   | Precision  | Recall     | F1-Score   | AUC-ROC    |
| ---------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| TF-IDF + LR (Baseline) | 93.65%     | 94.22%     | 92.98%     | 93.60%     | 97.74%     |
| **TGNN**               | **96.10%** | **94.29%** | **97.63%** | **95.93%** | **99.33%** |
| Improvement            | +2.45%     | +0.07%     | +4.65%     | +2.33%     | +1.59%     |

### Diffusion Prediction Results

| Metric  | Value | Description                |
| ------- | ----- | -------------------------- |
| Hit@1   | 20.0% | Top-1 prediction accuracy  |
| Hit@5   | 56.0% | Top-5 prediction accuracy  |
| Hit@10  | 76.0% | Top-10 prediction accuracy |
| MRR     | 37.3% | Mean Reciprocal Rank       |
| Jaccard | 17.2% | Set overlap similarity     |

### Moderation Strategy Effectiveness

| Strategy          | Effectiveness Score | Hate Reduction |
| ----------------- | ------------------- | -------------- |
| Content Removal   | 9.98                | 11.1%          |
| User Banning      | 0.00                | 0.0%           |
| Subreddit Banning | 0.00                | 0.0%           |
| Combined Strategy | 4.18                | 4.7%           |

## Configuration

The system uses YAML configuration files to manage parameters. Key configuration options include:

```yaml
# Example configuration
data_prep:
  max_samples_per_file: 50000
  target_dataset_size: 2000

bert:
  model_name: "distilbert-base-uncased"
  batch_size: 32
  device: "cuda"

tgnn:
  hidden_dim: 128
  num_epochs: 100
  learning_rate: 0.001

diffusion:
  k_values: [1, 5, 10, 20]
  prediction_window: 24
```

## Technical Highlights

### Innovation

- **Temporal Graph Neural Networks**: First application to hate speech diffusion prediction
- **Multi-modal Feature Fusion**: Combines text, user behavior, and network structure features
- **End-to-end Learning**: Complete pipeline from raw data to final predictions

### Engineering

- **Modular Design**: Independent scripts for each component
- **GPU Acceleration**: CUDA support for efficient training and inference
- **Configuration-driven**: Flexible parameter management through YAML files

### Evaluation

- **Multi-dimensional Metrics**: Classification, ranking, and diffusion prediction evaluation
- **Realistic Validation**: Results validated against social network characteristics
- **Comprehensive Visualization**: Rich charts and interactive visualizations

## Usage Examples

### Running Individual Components

```bash
# Data preprocessing only
python scripts/00_data_preparation.py --config configs/exp_small.yaml

# Baseline model comparison
python scripts/baseline_tfidf_lr.py --config configs/exp_gpu_7days.yaml

# Diffusion prediction with custom parameters
python scripts/04_diffusion_prediction.py --config configs/exp_gpu_7days.yaml
```

### Custom Configuration

Create your own configuration file by modifying the existing templates:

```yaml
# custom_config.yaml
data_prep:
  max_samples_per_file: 10000
  target_dataset_size: 1000

bert:
  model_name: "bert-base-uncased"
  batch_size: 16
```

## Output Files

The system generates various output files in the `artifacts/` and `figures/` directories:

- `tgnn_metrics.json`: Model performance metrics
- `diffusion_prediction_results.json`: Diffusion prediction results
- `moderation_simulation_results.json`: Moderation strategy evaluation
- `comprehensive_evaluation_report.json`: Overall system evaluation
- Various visualization files in PNG and HTML formats

## Performance Optimization

### GPU Acceleration

- Ensure CUDA is properly installed
- Use appropriate batch sizes for your GPU memory
- Monitor GPU utilization during training

### Memory Management

- Adjust `max_samples_per_file` based on available RAM
- Use smaller batch sizes for memory-constrained environments
- Consider data sampling for large datasets
