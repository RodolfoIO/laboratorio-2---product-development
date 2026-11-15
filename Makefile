.PHONY: install test lint format check compose-up compose-down acceptance

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check src tests dags migrations

format:
	uv run ruff format src tests dags migrations

check: lint test

compose-up:
	docker compose up --build

compose-down:
	docker compose down

acceptance:
	uv run python scripts/acceptance.py
