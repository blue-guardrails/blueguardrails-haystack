.PHONY: lint format format-check types unit integration quality test

lint:
	uv run ruff check .

format:
	uv run ruff check --fix --select I .
	uv run ruff format .

format-check:
	uv run ruff format --check .

types:
	uv run pyright

unit:
	uv run pytest tests/unit

integration:
	uv run --extra integration pytest -o addopts="" tests/integration -m "not benchmark"

quality: lint format-check types

test: unit integration
