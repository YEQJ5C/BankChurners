.PHONY: init check lint typecheck add-data prepare features train evaluate test run pipeline

init:
	pip install -r requirements.txt

check:
	python run_check.py

lint:
	flake8 src tests

typecheck:
	pyright src

add-data:
	dvc add data/raw/BankChurners.csv

prepare: check
	python -m src.prepare

features: prepare
	python -m src.features

train: features
	python -m src.train

evaluate: train
	python -m src.evaluate

pipeline: evaluate
	@echo "A teljes MLOps pipeline sikeresen lefutott!"

test:
	pytest --cov=src

run:
	uvicorn src.api:app --reload