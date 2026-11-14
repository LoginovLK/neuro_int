# neuro_int v2 — index-finger pointing -> cursor. One process, no docker.

.DEFAULT_GOAL := help
VENV_DIR := .venv-edge
VENV_PY := $(VENV_DIR)/Scripts/python.exe
# ":=" (not "?=") — an empty PY from the environment must not win.
PY := python
ENV := PYTHONPATH=src

.PHONY: help run headless smoke diag test lint clean

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

run: ## control the cursor with the index finger (debug window; q quits)
	$(ENV) $(VENV_PY) -m neuroint_cursor

headless: ## same, without the debug window
	$(ENV) $(VENV_PY) -m neuroint_cursor --no-window

smoke: ## 12s self-check: camera + GPU + detection (mouse untouched)
	$(ENV) $(VENV_PY) -m neuroint_cursor --no-window --dry --smoke 12

diag: ## 8s capture with your hand at the camera -> captures/diag_hands.png
	$(ENV) $(VENV_PY) tools/diag.py

test: ## unit tests (no camera needed)
	$(ENV) $(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

clean: ## remove caches
	@rm -rf src/neuroint_cursor/__pycache__ tests/__pycache__ .pytest_cache 2>/dev/null || true
