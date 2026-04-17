.PHONY: install train evaluate optimize serve dashboard test lint format clean help

# ─── Environment ──────────────────────────────────────────────
install: ## Install all dependencies
	pip install -r requirements.txt

install-dev: ## Install with dev dependencies
	pip install -r requirements.txt
	pre-commit install

# ─── Training ─────────────────────────────────────────────────
train: ## Train the model (use MODEL=lstm|transformer)
	python scripts/train.py --config configs/config.yaml --model $(or $(MODEL),transformer)

train-lstm: ## Train LSTM baseline
	python scripts/train.py --config configs/config.yaml --model lstm

train-transformer: ## Train Transformer model
	python scripts/train.py --config configs/config.yaml --model transformer

finetune: ## Demonstrate LLM Fine-Tuning (LoRA/PEFT)
	PYTHONPATH=. python scripts/finetune_llm.py

# ─── Evaluation ───────────────────────────────────────────────
evaluate: ## Evaluate model performance
	python scripts/evaluate.py --config configs/config.yaml

# ─── Optimization ─────────────────────────────────────────────
optimize: ## Run inference optimization pipeline
	python scripts/optimize.py --config configs/config.yaml

benchmark: ## Run inference benchmarks
	python scripts/optimize.py --config configs/config.yaml --benchmark-only

# ─── Serving ──────────────────────────────────────────────────
serve: ## Start FastAPI server
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard: ## Start Streamlit dashboard
	streamlit run dashboard/app.py --server.port 8501

# ─── Testing ──────────────────────────────────────────────────
test: ## Run all tests
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast: ## Run tests excluding slow ones
	pytest tests/ -v -m "not slow" --cov=src

# ─── Code Quality ─────────────────────────────────────────────
lint: ## Run linters
	ruff check src/ tests/
	mypy src/

format: ## Format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

# ─── MLflow ───────────────────────────────────────────────────
mlflow-ui: ## Start MLflow UI
	mlflow ui --port 5000

# ─── Docker ───────────────────────────────────────────────────
docker-build: ## Build Docker image
	docker-compose build

docker-up: ## Start all services
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

# ─── Utilities ────────────────────────────────────────────────
clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info mlruns/

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
