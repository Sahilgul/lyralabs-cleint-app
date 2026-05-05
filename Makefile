.PHONY: help gen-key dev dev-api dev-worker dev-worker-socket dev-redis migrate test test-regression test-inference test-inference-one eval-arlo eval-arlo-model eval-arlo-case eval-arlo-ghl eval-arlo-ghl-model test-watch test-coverage lint type fmt clean

help:
	@echo "Docker dev:    dev"
	@echo "Native dev:    dev-redis (one-time), then in two terminals: dev-api, dev-worker"
	@echo "Other:         gen-key, migrate, test, test-regression, test-inference, test-watch, test-coverage, lint, type, fmt, clean"
	@echo ""
	@echo "Inference benchmark (speed/cost, real API calls, needs keys in .env):"
	@echo "  make test-inference              # all 6 models, comparison table"
	@echo "  make test-inference-one M='DeepSeek V4-Pro'  # single model"
	@echo ""
	@echo "ARLO task eval (mock schemas, logs saved to logs/arlo-eval-*):"
	@echo "  make eval-arlo                   # all 6 models × 15 cases → summary.json"
	@echo "  make eval-arlo-model M='MiniMax M2.7'          # one model × all cases"
	@echo "  make eval-arlo-case  C='ghl-contact-search'   # one case × all models"
	@echo ""
	@echo "ARLO × GHL live eval (real MCP calls, requires GHL_EVAL_TOKEN in .env):"
	@echo "  make eval-arlo-ghl               # all 6 models × 11 cases, real CRM data"
	@echo "  make eval-arlo-ghl-model M='MiniMax M2.7'     # one model, real GHL data"
	@echo ""
	@echo "The admin UI lives in a separate repo (../lyralabs-admin-ui)."
	@echo "Run \`npm run dev\` there for the Vite dev server on :5173."

gen-key:
	@python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# --- Docker dev (everything in containers) -----------------------------------
dev:
	docker compose -f infra/docker-compose.yml up --build

# --- Native dev (no Docker) --------------------------------------------------
# Postgres lives wherever DATABASE_URL points (Supabase, Neon, or local).
# Redis runs natively via systemd (one-time start).
# Run dev-api in one terminal and dev-worker in another.
dev-redis:
	sudo systemctl start redis-server
	@redis-cli ping

dev-api:
	PYTHONPATH=packages:. .venv/bin/uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

dev-worker:
	PYTHONPATH=packages:. .venv/bin/python -m arq apps.worker.arq_app.WorkerSettings

dev-socket:
	PYTHONPATH=packages:. .venv/bin/python -m apps.socket_listener.main

migrate:
	PYTHONPATH=packages:. .venv/bin/alembic upgrade head

test:
	APP_ENV=test PYTHONPATH=packages:. .venv/bin/pytest tests/unit -q

test-regression:
	APP_ENV=test PYTHONPATH=packages:. .venv/bin/pytest tests/regression/regression_test1.py tests/regression/regression_test2.py tests/regression/regression_test3.py tests/regression/regression_test4.py tests/regression/regression_test5.py tests/regression/regression_test6.py tests/regression/regression_test7.py tests/regression/regression_test8.py tests/regression/regression_test9.py tests/regression/regression_test10.py tests/regression/regression_test11.py -v

# Live inference benchmark — hits real provider APIs, requires keys in .env.
# Prints latency, tokens/s, cost, and a response snippet for every model.
test-inference:
	PYTHONPATH=packages:. .venv/bin/pytest tests/inference/test_model_inference.py::test_inference_benchmark -v -s -m live

# Run a single model by name: make test-inference-one M="DeepSeek V4-Pro"
test-inference-one:
	PYTHONPATH=packages:. .venv/bin/pytest tests/inference/test_model_inference.py -k "$(M)" -v -s -m live

# ---------------------------------------------------------------------------
# ARLO task eval — quality on real GHL + Slack tasks, JSON logs in logs/
# ---------------------------------------------------------------------------

# All 6 models × all 15 cases (mock schemas). Logs → logs/arlo-eval-<timestamp>/
eval-arlo:
	PYTHONPATH=packages:. .venv/bin/pytest tests/eval/test_arlo_eval.py::test_arlo_eval_all -v -s -m live

# One model × all 15 cases: make eval-arlo-model M="MiniMax M2.7"
eval-arlo-model:
	PYTHONPATH=packages:. .venv/bin/pytest tests/eval/test_arlo_eval.py::test_arlo_eval_model -k "$(M)" -v -s -m live

# One case × all 6 models: make eval-arlo-case C="ghl-contact-search"
eval-arlo-case:
	PYTHONPATH=packages:. .venv/bin/pytest tests/eval/test_arlo_eval.py::test_arlo_eval_case -k "$(C)" -v -s -m live

# ---------------------------------------------------------------------------
# ARLO × GHL LIVE eval — real MCP tool calls, real CRM data
# Requires GHL_EVAL_TOKEN and GHL_EVAL_LOCATION_ID in .env
# (GHL Private Integration Token, read-only scopes, no mutations)
# ---------------------------------------------------------------------------

# All 6 models × all 11 GHL cases. Logs → logs/arlo-ghl-live-<timestamp>/
eval-arlo-ghl:
	PYTHONPATH=packages:. .venv/bin/pytest tests/eval/test_arlo_ghl_live.py::test_arlo_ghl_live_all -v -s -m live

# One model with real GHL: make eval-arlo-ghl-model M="MiniMax M2.7"
eval-arlo-ghl-model:
	PYTHONPATH=packages:. .venv/bin/pytest tests/eval/test_arlo_ghl_live.py::test_arlo_ghl_live_model -k "$(M)" -v -s -m live

test-watch:
	APP_ENV=test PYTHONPATH=packages:. pytest tests/unit -q --tb=short --maxfail=1 -x

test-coverage:
	APP_ENV=test PYTHONPATH=packages:. pytest tests/unit \
		--cov=packages/lyra_core --cov=apps --cov-report=term-missing

lint:
	ruff check .

type:
	mypy packages apps

fmt:
	ruff format .
	ruff check --fix .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
