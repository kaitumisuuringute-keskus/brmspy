# brmspy Makefile - Modern Python Development Workflow

.PHONY: help install dev test lint format clean docs build dist upload-test upload

# Default target
help:
	@echo "brmspy - Modern Python Development Workflow"
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install package in development mode"
	@echo "  dev          Install with all development dependencies"
	@echo "  test         Run tests with pytest"
	@echo "  lint         Run linters (ruff, mypy)"
	@echo "  format       Format code with black and isort"
	@echo "  clean        Remove build artifacts and caches"
	@echo "  docs         Build documentation (if using Sphinx)"
	@echo "  build        Build distribution packages"
	@echo "  dist         Alias for build"
	@echo "  upload-test  Upload to TestPyPI"
	@echo "  upload       Upload to PyPI"

# Install package in development mode
install:
	pip install -e .

# Install with all development dependencies
dev:
	pip install -e ".[all]"

# Run tests
test:
	pytest -v --cov=brmspy --cov-report=term-missing

# Run linters
lint:
	@echo "Running ruff..."
	ruff check brmspy/
	@echo "Running mypy..."
	mypy brmspy/

# Format code
format:
	@echo "Formatting with black..."
	black brmspy/ examples/
	@echo "Sorting imports with isort..."
	isort brmspy/ examples/
	@echo "Done!"

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned build artifacts"

# Build documentation (placeholder for future Sphinx setup)
docs:
	@echo "Documentation build not yet configured"
	@echo "To set up: install sphinx and configure docs/"

# Build distribution packages
build: clean
	python -m build

# Alias for build
dist: build

# Upload to TestPyPI
upload-test: dist
	twine upload --repository testpypi dist/*

# Upload to PyPI
upload: dist
	twine upload --repository pypi dist/*