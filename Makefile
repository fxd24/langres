.PHONY: help build clean publish test test-full

help:
	@echo "Available commands:"
	@echo "  make build     - Build the package"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make publish   - Publish package to PyPI"
	@echo "  make test      - Run fast tests (no slow/integration)"
	@echo "  make test-full - Run all tests with coverage"

build:
	uv build

clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

publish: build
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found"; \
		exit 1; \
	fi
	@source .env && uv publish

test:
	uv run pytest -m "not slow and not integration"

test-full:
	uv run pytest --cov --cov-report=term-missing
