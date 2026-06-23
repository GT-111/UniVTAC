.PHONY: lint format check test smoke docker-build docker-push clean help

VENV := .venv
UV := uv

# ── Code quality ───────────────────────────────────────────────────────

lint:
	$(UV) run ruff check --fix

format:
	$(UV) run ruff format

check:
	$(UV) run ruff check
	$(UV) run ruff format --check

# ── Testing ────────────────────────────────────────────────────────────

test:
	$(UV) run pytest

smoke:
	$(UV) run python -c "import isaacsim; import isaaclab; import curobo; print('Core OK')"
	$(UV) run univtac list tasks
	$(UV) run univtac list policies

# ── Docker ─────────────────────────────────────────────────────────────

docker-build:
	bash docker/build.sh

docker-push:
	bash docker/push.sh

# ── Cleanup ────────────────────────────────────────────────────────────

clean:
	rm -rf build/ dist/ *.egg-info src/univtac.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

# ── Help ───────────────────────────────────────────────────────────────

help:
	@echo "make lint          — auto-fix lint issues"
	@echo "make format        — auto-format code"
	@echo "make check         — lint + format check (CI)"
	@echo "make test          — run pytest"
	@echo "make smoke         — quick smoke test"
	@echo "make docker-build  — build all Docker images"
	@echo "make docker-push   — push images to GHCR"
	@echo "make clean         — remove build artifacts"
