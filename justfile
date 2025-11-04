set shell := ['bash', '-ceuo', 'pipefail']

@default: lint test

@lint:
    ruff check
    ruff format --check
    ruff check
    pyright

@test:
    pytest
