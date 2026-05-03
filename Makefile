.PHONY: help gen-key dev dev-api dev-worker dev-redis migrate test test-regression test-watch test-coverage lint type fmt clean

help:
	@echo "Docker dev:    dev"
	@echo "Native dev:    dev-redis (one-time), then in two terminals: dev-api, dev-worker"
	@echo "Other:         gen-key, migrate, test, test-watch, test-coverage, lint, type, fmt, clean"
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

migrate:
	PYTHONPATH=packages:. .venv/bin/alembic upgrade head

test:
	APP_ENV=test PYTHONPATH=packages:. .venv/bin/pytest tests/unit -q

test-regression:
	APP_ENV=test PYTHONPATH=packages:. .venv/bin/pytest tests/regression/regression_test1.py tests/regression/regression_test2.py tests/regression/regression_test3.py -v

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
