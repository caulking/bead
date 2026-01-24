# Installation

## Requirements

- Python 3.13 or higher
- Operating Systems: macOS, Linux, Windows (WSL recommended)

## Install from PyPI

Install the latest stable release:

```bash
uv pip install bead
```

## Install from Source

For the latest development version:

```bash
git clone https://github.com/aaronstevenwhite/bead.git
cd bead
uv sync
```

## Development Installation

For contributing to bead, install with all development dependencies:

```bash
uv sync --all-extras
```

This installs:
- `dev`: testing and linting tools (pytest, ruff, pyright)
- `api`: model adapters (OpenAI, Anthropic, Google, HuggingFace)
- `training`: active learning dependencies (PyTorch, transformers)

## Verify Installation

Check that bead is installed correctly:

```bash
uv run python -c "import bead; print(bead.__version__)"
```

Or use the CLI:

```bash
uv run bead --version
```

## Optional Dependencies

Install specific dependency groups as needed:

```bash
# HuggingFace models for template filling and active learning
uv sync --extra api

# Active learning with PyTorch
uv sync --extra training

# All dependencies
uv sync --all-extras
```

## Troubleshooting

### Python Version

Verify you have Python 3.13+:

```bash
python --version
```

If not, install from [python.org](https://www.python.org/downloads/) or use pyenv:

```bash
pyenv install 3.13.0
pyenv local 3.13.0
```

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'bead'`
**Solution**: Ensure you are using `uv run` to execute Python commands.

**Issue**: `ImportError` for optional dependencies
**Solution**: Install the required extra, e.g., `uv sync --extra api`

## Next Steps

Continue to the [Quick Start](quickstart.md) guide for a complete tutorial.
