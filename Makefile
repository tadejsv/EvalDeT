.PHONY: lint
lint:
	ruff .
	black --check .
	isort --check-only .
	mypy .

.PHONY: format
format:
	black .
	isort .
