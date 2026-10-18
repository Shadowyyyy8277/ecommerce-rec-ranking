# E-Commerce Recommendation Multi-Channel Recall & CTR Ranking System

[**English**](README_EN.md) | [**中文版**](README.md)

[![CI](https://github.com/Shadowyyyy8277/ecommerce-rec-ranking/actions/workflows/ci.yml/badge.svg)](https://github.com/Shadowyyyy8277/ecommerce-rec-ranking/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)

An end-to-end industrial-grade e-commerce recommendation system covering data simulation, leakage-free feature engineering, multi-channel concurrent recall, CTR ranking model comparison (LR, LightGBM, DeepFM), offline top-K evaluation, MLflow tracking, dynamic model registry, sub-millisecond FastAPI serving, and an interactive Streamlit dashboard.

---

## Key Highlights

- **End-to-End Pipeline**: Full recommendation lifecycle from raw behavior simulation to online serving and visual demo.
- **Concurrent Multi-Channel Recall**: Parallel execution of **Popular Recall** (behavior-weighted), **ItemCF** (co-occurrence with position and popularity discount), and **Item2Vec** (PyTorch Skip-gram with negative sampling), with adaptive confidence-based deduplication.
- **Leakage-Free Feature Engineering**: Strict temporal split ($Train < Valid < Test$); cumulative historical statistics computed strictly on past events with zero label leakage.
- **CTR Ranking & Deep Learning**: Benchmarks **Logistic Regression**, **LightGBM**, and **DeepFM** (first-order linear + second-order FM interactions + deep MLP with **Cosine Annealing LR Scheduler**).
- **Pluggable Model Registry**: Decoupled registry design (`src/models/registry.py`) allowing seamless plug-and-play of new ranking algorithms without modifying downstream serving or evaluation pipelines.
- **Comprehensive Evaluation**: Evaluates binary classification metrics (**AUC**, **LogLoss**), Top-K ranking metrics (**Precision@K**, **Recall@K**, **NDCG@K**, **HitRate@K**, **MRR@K**), along with inference latency and model memory footprint.
- **High-Throughput Online Serving**: FastAPI service with Pydantic v2 validation, LRU cache with copy protection, and automatic cold-start popular fallback.
- **Production-Ready Engineering**: Modern packaging (`pyproject.toml`), `Makefile` commands, Docker & Docker Compose orchestration, and 22 automated test suites running via GitHub Actions CI.

---

## Architecture

```
User Interactions (Simulated E-Commerce Logs)
       │
       ▼
Time-Series Split & Feature Engineering (Cumulative Statistics)
       │
       ├──► Multi-Channel Concurrent Recall (ThreadPoolExecutor)
       │       ├── Popular Recall (Weighted Behavior)
       │       ├── ItemCF (Time-Decay Co-occurrence)
       │       └── Item2Vec (Skip-Gram Negative Sampling)
       │               │
       │               ▼
       │       Merged & Deduplicated Candidates (Top-N per user)
       │               │
       ▼               ▼
CTR Ranking Models (Model Registry Pattern)
       ├── Logistic Regression (Baseline)
       ├── LightGBM (Gradient Boosted Trees - Best Performer)
       └── DeepFM (PyTorch FM + DNN + CosineAnnealingLR)
               │
               ▼
Offline Evaluation & Benchmarking (AUC, LogLoss, NDCG, HitRate, MRR)
               │
               ▼
Model Registry (`artifacts/model_registry.json`) & MLflow Tracking
               │
       ┌───────┴────────────────────────┐
       ▼                                ▼
FastAPI Online Serving          Streamlit Visual Dashboard
(http://localhost:8000)          (http://localhost:8501)
```

---

## Benchmark Results

### Validation Set Comparison

| Model | Valid AUC | Valid LogLoss | Status |
|---|---:|---:|:---:|
| **LightGBM** (Winner) | **0.5784** | **0.3917** | **Selected for Serving** |
| Logistic Regression | 0.5666 | 0.3955 | Baseline |
| DeepFM | 0.5417 | 0.4200 | Neural Model |

### Test Set Metrics (Selected Model: LightGBM)

| Metric | Score | Notes |
|---|---:|---|
| **Test AUC** | **0.5842** | Classification discriminative ability |
| **Test LogLoss** | **0.3842** | Cross-entropy probability calibration |
| **HitRate@5** | 0.0379 | Top-5 recommendation hit rate |
| **HitRate@10** | 0.0966 | Top-10 recommendation hit rate |
| **HitRate@20** | 0.2034 | Top-20 recommendation hit rate |
| **NDCG@20** | 0.0578 | Position-discounted ranking quality |
| **Inference Latency** | **0.0368 ms** | Sub-millisecond CPU per-sample prediction |
| **Model Size** | **0.37 MB** | Compact deployment footprint |

---

## Project Structure

```text
ecommerce-rec-ranking
├── .github/workflows/ci.yml          # GitHub Actions CI pipeline
├── app
│   └── streamlit_app.py              # Streamlit interactive web dashboard
├── artifacts
│   ├── models/                       # Serialized model weights (.joblib, .pt)
│   ├── recall/                       # Multi-channel candidate sets
│   └── model_registry.json           # Selected production model metadata
├── configs
│   └── config.yaml                   # Central project configuration
├── data
│   ├── raw/                          # Raw simulated user interactions
│   └── processed/                    # Train, valid, and test feature splits
├── docs
│   ├── project_report.md             # In-depth architectural report (Chinese)
│   ├── metrics.json                  # Machine-readable evaluation report
│   └── screenshots/                  # Application UI & MLflow screenshots
├── notebooks
│   └── 01_data_exploration.ipynb     # In-depth exploratory data analysis (EDA)
├── reports
│   ├── metrics/                      # Offline evaluation tables & comparisons
│   └── project_summary.md            # Resume bullet points & interview questions
├── scripts
│   ├── run_demo_data.py              # Generate synthetic user behavior data
│   ├── run_data_pipeline.py          # Feature extraction & dataset splitting
│   ├── run_recall.py                 # Concurrent multi-channel recall
│   ├── run_train_all.py              # Train LR, LightGBM, DeepFM & update registry
│   └── run_evaluate.py               # Offline Top-K & CTR evaluation
├── src
│   ├── data/                         # Data generator & loading orchestration
│   ├── features/                     # Feature engineering & shared encoders
│   ├── recall/                       # Popular, ItemCF, Item2Vec recall engines
│   ├── models/                       # Model definitions, trainers, & registry
│   ├── evaluation/                   # Offline metrics (AUC, NDCG, MRR, HitRate)
│   ├── serving/                      # FastAPI online inference application
│   └── utils/                        # Config cache, seed, logger, & MLflow helpers
├── tests/                            # 22 automated unit & integration tests
├── Dockerfile                        # Multi-stage production container image
├── docker-compose.yml                # API + Dashboard + MLflow service orchestration
├── Makefile                          # Unified build and automation commands
├── pyproject.toml                    # Standard packaging specification (PEP 621)
├── requirements.txt                  # Locked Python dependencies
└── README_EN.md                      # English documentation
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/Shadowyyyy8277/ecommerce-rec-ranking.git
cd ecommerce-rec-ranking

# Create conda environment
conda create -n rec-rank python=3.10 -y
conda activate rec-rank

# Editable install with development tools
pip install -e ".[dev]"
```

### 2. Run with Makefile (Recommended)

```bash
# View all available shortcuts
make help

# Run entire pipeline: data generation -> recall -> training -> evaluation
make pipeline

# Run the 22 automated tests
make test

# Start the FastAPI serving app (API docs at http://localhost:8000/docs)
make serve

# Start the Streamlit interactive dashboard (http://localhost:8501)
make dashboard
```

### 3. Run with Docker Compose

Spin up FastAPI, Streamlit Dashboard, and MLflow Server simultaneously:

```bash
docker compose up -d --build
```

Access endpoints:
- **FastAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
- **MLflow Tracking UI**: [http://localhost:5000](http://localhost:5000)

### 4. Step-by-Step Manual Scripts

```bash
# 1. Synthesize interactions and build feature splits
python scripts/run_demo_data.py
python scripts/run_data_pipeline.py

# 2. Execute multi-channel concurrent recall
python scripts/run_recall.py

# 3. Train all ranking models and pick best model
python scripts/run_train_all.py

# 4. Evaluate offline Top-K metrics
python scripts/run_evaluate.py
```

---

## API Reference

### Health Check: `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "lightgbm",
  "best_model": "lightgbm"
}
```

### Top-K Recommendation: `POST /recommend`

**Request:**
```json
{
  "user_id": "u_000001",
  "top_k": 5
}
```

**Response:**
```json
{
  "user_id": "u_000001",
  "recommendations": [
    {
      "item_id": "i_000284",
      "score": 0.4328,
      "reason": "Top-1 recommendation: predicted CTR of 43.3% by LightGBM model"
    },
    {
      "item_id": "i_000119",
      "score": 0.3872,
      "reason": "Top-2 recommendation: predicted CTR of 38.7% by LightGBM model"
    }
  ]
}
```

---

## Testing & Quality Assurance

Run the comprehensive test suite (22 unit & integration tests):

```bash
pytest
```

Tests cover:
- Pydantic request validation and fallback under cold-start / missing models
- Popular recall user history masking and custom weight assignment
- ItemCF co-occurrence similarity matrix construction
- Item2Vec Skip-gram embedding convergence
- Shared category encoders and numeric scaling
- Model registry pluggable dispatch and invalid model rejection
- Top-K metric computations (Precision, Recall, NDCG, HitRate, MRR)

---

## License

This project is licensed under the [MIT License](LICENSE).
