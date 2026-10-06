.PHONY: install lint typecheck test all train

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests scripts

typecheck:
	mypy src

test:
	pytest

all: lint typecheck test

train:
	python scripts/train.py
