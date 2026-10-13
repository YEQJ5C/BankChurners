.PHONY: init check lint typecheck add-data

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