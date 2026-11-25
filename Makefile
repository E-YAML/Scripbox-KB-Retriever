.PHONY: format lint lint-fix typecheck test test-cov check dev-install

format:
	ruff format .

lint:
	ruff check .

lint-fix:
	ruff check --fix .

typecheck:
	mypy scripbox_kb/ --ignore-missing-imports

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=scripbox_kb --cov-report=term-missing

check:
	ruff format --check .
	ruff check .
	mypy scripbox_kb/ --ignore-missing-imports
	pytest tests/ -v

dev-install:
	pip install -e ".[dev]"
