.PHONY: lint
lint:
	flake8 .
	black --check .
	isort --check-only .
	mypy .

.PHONY: format
format:
	black .
	isort .
