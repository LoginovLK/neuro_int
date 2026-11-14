# neuro_int — gesture control stack.
# Canonical workflow: `make rebuild` tears everything down, rebuilds and
# starts the full stack (brain container + host edge agent + UI).

.DEFAULT_GOAL := help
VENV_DIR := .venv-edge
VENV_PY := $(VENV_DIR)/Scripts/python.exe
# ":=" (not "?=") — an empty PY from the environment must not win.
PY := python
# make.exe does not extend PATH with Windows system dirs: full path needed.
POWERSHELL := C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe

.PHONY: help rebuild down build up edge-start edge-stop restart-edge install-edge run ui test lint format probe smoke logs ps clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

rebuild: down build up ## Full stack rebuild: down -> build -> up (brain + edge + UI)

down: ## Stop everything: edge agent + brain container
	@$(MAKE) edge-stop
	@timeout 60 docker compose down --remove-orphans --timeout 10 \
	|| (docker rm -f $$(docker compose ps -q) 2>/dev/null; echo ">> container force-removed (was stuck)")
	@echo ">> stack stopped"

build: ## Build the brain image (cached layers, uv cache mounts)
	@timeout 600 docker compose build

up: install-edge edge-start ## Start everything and WAIT until edge (model + UI) is fully up
	docker compose up -d --wait
	@echo ">> brain healthy on 127.0.0.1:$${NEUROINT_WS_PORT:-8765}"
	@echo ">> waiting for edge agent (model load + UI)..."
	@ok=0; for i in $$(seq 1 90); do \
	  code=$$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$${NEUROINT_UI_PORT:-8756}/ 2>/dev/null); \
	  if [ "$$code" = "200" ]; then ok=1; echo ">> edge ready (model loaded, UI up)"; break; fi; \
	  sleep 1; \
	done; \
	if [ "$$ok" != "1" ]; then echo ">> edge did not come up in 90s — check make logs"; exit 1; fi

install-edge: ## Create edge venv and install agent (+pseyepy, +neural super-res) on first run; refresh extras later
	@if [ ! -f "$(VENV_PY)" ]; then \
	  echo ">> one-time setup: creating edge venv (compiles pseyepy, needs MSVC)..."; \
	  $(PY) -m venv $(VENV_DIR) \
	  && $(VENV_PY) -m pip install --upgrade pip --quiet \
	  && $(VENV_PY) -m pip install -e src/protocol -e "src/edge[pseye]" --quiet; \
	else \
	  echo ">> edge venv ready"; \
	fi
	@$(VENV_PY) -m pip uninstall -y opencv-python --quiet 2>/dev/null || true
	@$(VENV_PY) -m pip install opencv-contrib-python --quiet
	@$(VENV_PY) -m pip install onnxruntime-directml --quiet \
	|| echo ">> onnxruntime-directml unavailable: realesrgan backend will stay off (fsrnn works anyway)"
	@if [ ! -f "models/palm_detection_full_inf_post_192x192.onnx" ] || [ ! -f "models/hand_landmark_sparse_Nx3x224x224.onnx" ]; then \
	  mkdir -p models \
	  && echo ">> downloading hand detection models (PINTO ONNX, ~15MB, one-time)..." \
	  && curl -fsSL --retry 3 --connect-timeout 20 -o models/palm_detection_full_inf_post_192x192.onnx \
	     "https://raw.githubusercontent.com/PINTO0309/hand-gesture-recognition-using-onnx/main/model/palm_detection/palm_detection_full_inf_post_192x192.onnx" \
	  && curl -fsSL --retry 3 --connect-timeout 20 -o models/hand_landmark_sparse_Nx3x224x224.onnx \
	     "https://raw.githubusercontent.com/PINTO0309/hand-gesture-recognition-using-onnx/main/model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx" \
	  && echo ">> models downloaded"; \
	else \
	  echo ">> hand detection models ready"; \
	fi

# PID-file managed background agent. Recipes run through bash (git bash in
# PATH): outer single quotes protect PowerShell "$$" vars from bash; inner
# PowerShell strings use double quotes (no "$" inside them).
# Kill filter matches BOTH python.exe and pythonw.exe (a zombie from another
# session must die too); the UI port holder is killed before starting.
edge-start: ## Start edge agent detached (kills any old instance + frees UI port first)
	@$(POWERSHELL) -NoProfile -Command 'if (-not (Test-Path ".venv-edge/Scripts/pythonw.exe")) { Write-Error "missing venv: run make install-edge"; exit 1 }; Get-CimInstance Win32_Process | Where-Object { $$_.Name -match "python" -and $$_.CommandLine -match "neuroint_edge" } | ForEach-Object { Stop-Process -Id $$_.ProcessId -Force -ErrorAction SilentlyContinue }; Get-NetTCPConnection -LocalPort 8756 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $$_.OwningProcess -Force -ErrorAction SilentlyContinue }; if (-not (Test-Path "logs")) { New-Item -ItemType Directory "logs" | Out-Null }; $$p = Start-Process -WindowStyle Hidden -FilePath ".venv-edge/Scripts/pythonw.exe" -ArgumentList "-m","neuroint_edge.main" -RedirectStandardOutput "logs/edge.out.log" -RedirectStandardError "logs/edge.err.log" -PassThru; Set-Content -Path ".edge.pid" -Value $$p.Id; Write-Output (">> edge started (pid " + $$p.Id + "), logs: logs/edge.out.log")'

edge-stop: ## Stop ALL edge agents (any python flavor) and free the UI port
	@$(POWERSHELL) -NoProfile -Command 'Get-CimInstance Win32_Process | Where-Object { $$_.Name -match "python" -and $$_.CommandLine -match "neuroint_edge" } | ForEach-Object { try { Stop-Process -Id $$_.ProcessId -Force -ErrorAction Stop; Write-Output ">> edge stopped (pid $($$_.ProcessId))" } catch { Write-Error "cannot stop pid: $$_"; exit 1 } }; Get-NetTCPConnection -LocalPort 8756 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $$_.OwningProcess -Force -ErrorAction SilentlyContinue }; Remove-Item ".edge.pid" -ErrorAction SilentlyContinue; Write-Output ">> no edge agents left"'

restart-edge: edge-stop edge-start ## Restart only the edge agent (applies edge code edits)

run: ## Run edge agent in foreground (Ctrl+C stops; for debugging)
	$(VENV_PY) -m neuroint_edge.main

ui: ## Open the gesture UI in the browser (waits until the edge is ready)
	@echo ">> waiting for edge UI to come up..."
	@ok=0; for i in $$(seq 1 90); do \
	  code=$$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$${NEUROINT_UI_PORT:-8756}/ 2>/dev/null); \
	  if [ "$$code" = "200" ]; then ok=1; echo ">> UI ready, opening browser"; break; fi; \
	  sleep 1; \
	done; \
	if [ "$$ok" != "1" ]; then echo ">> edge UI did not come up in 90s — check make logs"; exit 1; fi; \
	cmd //c start "" "http://127.0.0.1:$${NEUROINT_UI_PORT:-8756}"

test: ## Run unit + integration tests
	$(PY) -m pytest -q

lint: ## Ruff lint (includes security rules)
	$(PY) -m ruff check src tests tools

format: ## Ruff format all code
	$(PY) -m ruff format src tests tools

probe: ## Camera probe: save sample shots, print tuning metrics (captures/)
	$(VENV_PY) -m neuroint_edge.probe

tune: ## Interactive camera tuner (arrows = gain/exposure; stop edge first)
	$(VENV_PY) -m neuroint_edge.tune

smoke: ## Smoke test: connect to running brain, verify protocol round-trip
	$(VENV_PY) tools/smoke_test.py

logs: ## Tail edge agent logs (use `make ps` for container status)
	tail -f logs/edge.out.log logs/edge.err.log

ps: ## Show stack status (container + edge agent)
	docker compose ps
	@$(POWERSHELL) -NoProfile -Command 'if (Test-Path ".edge.pid") { $$p = Get-Process -Id (Get-Content ".edge.pid") -ErrorAction SilentlyContinue; if ($$p) { Write-Output "edge: running" } else { Write-Output "edge: stopped (stale pid)" } } else { Write-Output "edge: not started" }'

clean: ## Remove venv, pid file, logs and local artifacts
	rm -rf $(VENV_DIR) .edge.pid logs captures .pytest_cache .ruff_cache
