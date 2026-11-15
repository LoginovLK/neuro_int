# neuro_int v2 — index-finger pointing -> cursor. One process, no docker.

.DEFAULT_GOAL := help
VENV_DIR := .venv-edge
VENV_PY := $(VENV_DIR)/Scripts/python.exe
# ":=" (not "?=") — an empty PY from the environment must not win.
PY := python
ENV := PYTHONPATH=src

.PHONY: help run headless smoke diag models lint clean

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

models: ## download model files into models/ (no-op once cached)
	$(PY) tools/fetch_models.py

run: models ## control the cursor with the index finger (debug window; q quits)
	$(ENV) $(VENV_PY) -m neuroint_cursor

headless: models ## same, without the debug window
	$(ENV) $(VENV_PY) -m neuroint_cursor --no-window

smoke: models ## 12s self-check: camera + GPU + detection (mouse untouched)
	$(ENV) $(VENV_PY) -m neuroint_cursor --no-window --dry --smoke 12

diag: ## 8s capture with your hand at the camera -> captures/diag_hands.png
	$(ENV) $(VENV_PY) tools/diag.py

lint:
	$(PY) -m ruff check src

clean: ## remove caches
	@rm -rf src/neuroint_cursor/__pycache__ 2>/dev/null || true
