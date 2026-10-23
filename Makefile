# Ex.: make index DATA_DIR=/caminho/para/i-care/training
DATA_DIR ?= $(NEUROICU_DATA_DIR)
MAX_HOUR ?= 72

.PHONY: install index test lint

install:
	pip install -e ".[dev]"

index:
	python -m neuroicu.index $(if $(DATA_DIR),--data-dir $(DATA_DIR)) --max-hour $(MAX_HOUR)

test:
	pytest -q

lint:
	ruff check src tests
