.PHONY: help build clean publish test test-full upgrade-deps audit scan-malware security

help:
	@echo "Available commands:"
	@echo "  make build         - Build the package"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make publish       - Publish package to PyPI"
	@echo "  make test          - Run fast tests (no slow/integration)"
	@echo "  make test-full     - Run all tests with coverage"
	@echo "  make upgrade-deps  - Roll the 7-day quarantine date forward and refresh uv.lock"
	@echo "  make audit         - Scan synced env for known CVEs (pip-audit)"
	@echo "  make scan-malware  - Scan dependencies for malicious indicators (guarddog)"
	@echo "  make security      - Run audit + scan-malware"

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

# Roll the quarantine window forward (today-7d on macOS) then re-solve all deps.
# Primary dev env is macOS; on Linux swap `date -v-7d` for `date -d '-7 days'`.
upgrade-deps:
	@NEW_DATE=$$(date -v-7d +%Y-%m-%d) && \
	python3 -c "import re, sys; \
	    text = open('pyproject.toml').read(); \
	    text = re.sub(r'exclude-newer = \"[0-9-]+\"', 'exclude-newer = \"' + '$$NEW_DATE' + '\"', text); \
	    open('pyproject.toml', 'w').write(text)" && \
	echo "exclude-newer set to $$NEW_DATE" && \
	uv lock --upgrade

audit:
	uv run pip-audit

scan-malware:
	uv export --no-hashes --format requirements-txt -o /tmp/langres-req.txt
	uv run guarddog pypi verify /tmp/langres-req.txt

security: audit scan-malware
