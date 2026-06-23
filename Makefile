# Convenience targets for the Data Analysis Chat Assistant.
# Everything runs against the project-local virtualenv in ./venv.

PYTHON ?= ./venv/bin/python
PIP := $(PYTHON) -m pip

.PHONY: help install check run lint format test eval clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime + dev dependencies into the venv (editable)
	$(PIP) install -e ".[dev]"

check: ## Verify Gemini + BigQuery connectivity (Phase 0 smoke test)
	$(PYTHON) scripts/check_access.py

run: ## Start the CLI chat assistant (added in Phase 2)
	$(PYTHON) -m assistant.cli

lint: ## Static checks with ruff
	$(PYTHON) -m ruff check src scripts

format: ## Auto-format and fix with ruff
	$(PYTHON) -m ruff format src scripts
	$(PYTHON) -m ruff check --fix src scripts

test: ## Run unit + component tests (added from Phase 5)
	$(PYTHON) -m pytest

eval: ## Run the golden evaluation harness (added in Phase 10)
	$(PYTHON) -m assistant.eval

clean: ## Remove caches and build artifacts
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info *.egg-info
	find . -type d -name __pycache__ -not -path './venv/*' -exec rm -rf {} +
