PYTHON := .venv/bin/python
PIP := .venv/bin/pip
MLFLOW := .venv/bin/mlflow
EXP ?= churn-exp

.PHONY: init data train evaluate predict test lint mlflow-ui clean

init:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) -m src.download_data --output data/raw.csv

train:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PYTHON) -m src.train --config configs/config.yaml

evaluate:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PYTHON) -m src.evaluate --config configs/config.yaml

predict:
	$(PYTHON) -m src.predict --config configs/config.yaml --input data/raw.csv --output artifacts/batch_predictions.csv

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests

mlflow-ui:
	$(MLFLOW) ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
