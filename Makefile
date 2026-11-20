# neuro_int v3 — gesture mouse (open hand + pinch). One process. git clone -> make run.

.DEFAULT_GOAL := help
VENV_DIR := .venv-edge
VENV_PY := $(VENV_DIR)/Scripts/python.exe
REQ_STAMP := $(VENV_DIR)/.req-stamp
# ":=" (not "?=") — an empty PY from the environment must not win.
PY := python
ENV := PYTHONPATH=src

.PHONY: help rebuild models run headless smoke diag lint clean

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

rebuild: ## (re)create the environment from scratch: venv + deps + models
	$(PY) -m venv $(VENV_DIR)
	$(VENV_PY) -m pip install --quiet --disable-pip-version-check -r requirements.txt
	$(PY) tools/fetch_models.py
	@echo "environment ready — make run"

# make run on a fresh clone sets everything up itself, no manual steps
$(VENV_PY):
	$(PY) -m venv $(VENV_DIR)

$(REQ_STAMP): requirements.txt | $(VENV_PY)
	$(VENV_PY) -m pip install --quiet --disable-pip-version-check -r requirements.txt
	@echo ok > $(REQ_STAMP)

models: ## download model files into models/ (no-op once cached)
	$(PY) tools/fetch_models.py

run: $(REQ_STAMP) models ## gesture mouse: open hand moves, pinch clicks (debug window; q quits)
	$(ENV) $(VENV_PY) -m neuroint_cursor

headless: $(REQ_STAMP) models ## same, without the debug window
	$(ENV) $(VENV_PY) -m neuroint_cursor --no-window

smoke: $(REQ_STAMP) models ## 12s self-check: camera + detector (mouse untouched)
	$(ENV) $(VENV_PY) -m neuroint_cursor --no-window --dry --smoke 12

diag: ## 8s capture with your hand at the camera -> captures/diag_hands.png
	$(ENV) $(VENV_PY) tools/diag.py

lint:
	$(PY) -m ruff check src

clean: ## remove caches
	@rm -rf src/neuroint_cursor/__pycache__ 2>/dev/null || true
