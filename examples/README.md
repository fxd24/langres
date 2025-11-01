# langres Examples

This directory contains example scripts demonstrating how to use the langres library.

## Available Examples

### Getting Started

- **verify_settings_and_clients.py** - Verify your environment configuration
  - Tests that Settings loads from .env correctly
  - Demonstrates LLM client creation (with/without Langfuse)
  - Tests wandb tracker initialization
  - Shows error handling for missing credentials
  - **Run this first to verify your setup!**

### Core API Examples

- **basic_usage.py** - Simple deduplication workflow using RapidfuzzModule
- **deduplication_with_blocker_optimization.py** - Complete optimization example
  - VectorBlocker for candidate generation
  - LLMJudgeModule with Azure OpenAI
  - BlockerOptimizer with Optuna + wandb
  - Demonstrates hyperparameter tuning
  - Requires: AZURE_API_*, WANDB_API_KEY, LANGFUSE_* env vars

## Setup

### 1. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
```

Required environment variables (only when using the respective services):
- `AZURE_API_ENDPOINT` - Azure OpenAI endpoint (for LLM calls)
- `AZURE_API_KEY` - Azure OpenAI API key (for LLM calls)
- `WANDB_API_KEY` - Weights & Biases key (for experiment tracking)
- `LANGFUSE_PUBLIC_KEY` - Langfuse public key (for LLM tracing)
- `LANGFUSE_SECRET_KEY` - Langfuse secret key (for LLM tracing)

**Note:** All environment variables are optional. Services only validate credentials when actually used.

### 2. Verify Setup

```bash
# Run the verification script to check your configuration
uv run python examples/verify_settings_and_clients.py
```

This will show which services are configured and test that everything works.

## Running Examples

```bash
# Verify setup first
uv run python examples/verify_settings_and_clients.py

# Run simple example (no API keys needed)
uv run python examples/basic_usage.py

# Run optimization example (requires API keys)
uv run python examples/deduplication_with_blocker_optimization.py
```

## Example Data

The `data/` directory contains sample datasets:
- `companies.json` - 100 company records with realistic duplicates
- `companies_labels.json` - Ground truth labels for evaluation

## Troubleshooting

If you see errors like "environment variable is required":
1. Make sure you've created a `.env` file (copy from `.env.example`)
2. Add the required API keys for the services you're using
3. Run `verify_settings_and_clients.py` to test your configuration

For Azure OpenAI errors:
- Verify `AZURE_API_BASE` is correctly formatted
- Check that `AZURE_API_KEY` is valid
- Ensure the deployment name (e.g., "gpt-5-mini") exists in your Azure resource
