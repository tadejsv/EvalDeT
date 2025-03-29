format:
    uv run ruff check --select I --fix .
    uv run ruff format .

lint:
    uv run ruff check .
    uv run ruff format --check

type-check:
    uv run mypy --strict .

check: lint type-check

test:
    uv run pytest -v

publish:
    uv build
    uv publish