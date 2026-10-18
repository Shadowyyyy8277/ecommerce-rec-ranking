.PHONY: help install data recall train evaluate serve dashboard test lint clean docker-build docker-up docker-down

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies and package in editable mode"
	@echo "  make data        - Generate synthetic data and run feature engineering pipeline"
	@echo "  make recall      - Run multi-channel recall (Popular, ItemCF, Item2Vec)"
	@echo "  make train       - Train all CTR ranking models (LR, LightGBM, DeepFM)"
	@echo "  make evaluate    - Evaluate best model offline and compute top-k metrics"
	@echo "  make pipeline    - Run full pipeline: data -> recall -> train -> evaluate"
	@echo "  make serve       - Start FastAPI recommendation serving API"
	@echo "  make dashboard   - Start interactive Streamlit demo application"
	@echo "  make test        - Run test suite"
	@echo "  make lint        - Run ruff code linter"
	@echo "  make clean       - Remove generated artifacts and caches"
	@echo "  make docker-up   - Start API, Dashboard, and MLflow using Docker Compose"

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) scripts/run_demo_data.py
	$(PYTHON) scripts/run_data_pipeline.py

recall:
	$(PYTHON) scripts/run_recall.py

train:
	$(PYTHON) scripts/run_train_all.py

evaluate:
	$(PYTHON) scripts/run_evaluate.py

pipeline: data recall train evaluate

serve:
	$(PYTHON) -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.port 8501

test:
	$(PYTHON) -m pytest -q --tb=short

lint:
	$(PYTHON) -m ruff check src scripts tests

clean:
	rm -rf data/raw/*.csv data/processed/*.csv artifacts/recall/*.csv artifacts/models/* reports/metrics/*.csv reports/metrics/*.json
