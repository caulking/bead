# Contributing to bead

First off, thank you for considering contributing to bead! It's people like you that make bead such a great tool for linguistic research.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Help](#getting-help)
- [What We're Looking For](#what-were-looking-for)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Your First Contribution](#your-first-contribution)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guides](#style-guides)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inclusive environment. By participating, you are expected to uphold this standard. Please be respectful, inclusive, and constructive in all interactions.

## Getting Help

- **Documentation**: Check [bead.readthedocs.io](https://bead.readthedocs.io) first
- **Discussions**: Use [GitHub Discussions](https://github.com/FACTSlab/bead/discussions) for questions and ideas
- **Issues**: Reserved for bug reports and feature requests with clear specifications

Please don't use the issue tracker for support questions. The discussions forum is a much better place to get help.

## What We're Looking For

bead is an open source project and we love to receive contributions from our community. There are many ways to contribute:

- **Bug reports**: Help us identify issues
- **Feature requests**: Suggest improvements with clear use cases
- **Documentation**: Fix typos, improve explanations, add examples
- **Code contributions**: Bug fixes, new features, performance improvements
- **Language examples**: Contribute research examples for new languages in the gallery

## Reporting Bugs

**Security vulnerabilities**: If you find a security vulnerability, do NOT open an issue. Email aaron.white@rochester.edu instead.

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

### Bug Report Template

```markdown
## Description
A clear and concise description of the bug.

## Steps to Reproduce
```python
from bead import ...

# Minimal code that reproduces the issue
```

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened, including full error traceback.

## Environment
- bead version: (run `bead --version`)
- Python version: (run `python --version`)
- OS: macOS / Linux / Windows
- Installation method: uv / pip
```

## Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use case**: Why is this feature needed? What problem does it solve?
- **Proposed solution**: How do you envision the feature working?
- **Alternatives considered**: Other approaches you've thought about
- **Code examples**: How the feature would be used

## Your First Contribution

Unsure where to begin? Look for issues labeled:

- `good first issue`: Simple issues suitable for newcomers
- `help wanted`: Issues where we'd appreciate community help

Working on your first Pull Request? Here are some resources:
- [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)
- [First Timers Only](https://www.firsttimersonly.com/)

## Development Setup

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) for package management
- Git

### Setup

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/bead.git
cd bead

# Add upstream remote
git remote add upstream https://github.com/FACTSlab/bead.git

# Install all dependencies
uv sync --all-extras

# Verify setup
uv run pytest tests/ -x
```

**Important**: Always use `uv run` to execute commands. Do not activate the virtual environment manually.

### Keeping Your Fork Updated

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the style guides below

3. **Write or update tests** for your changes

4. **Run all checks**:
   ```bash
   uv run ruff check .        # Lint
   uv run ruff format .       # Format
   uv run pyright             # Type check
   uv run pytest tests/       # Test
   ```

5. **Commit your changes** (see commit message guidelines below)

6. **Push to your fork** and open a Pull Request

### Pull Request Guidelines

- Fill in the PR template completely
- Link to any related issues using keywords (`Fixes #123`, `Closes #456`)
- Keep PRs focused on a single change
- Update documentation for any changed APIs
- Add tests for new functionality (target >90% coverage)
- Ensure all CI checks pass

### What to Expect

- A maintainer will review your PR, usually within a week
- You may be asked to make changes before merging
- Once approved, a maintainer will merge your PR

## Style Guides

### Python Code Style

- **Python version**: 3.13+ with modern syntax
- **Line length**: 88 characters
- **Formatter**: Ruff
- **Linter**: Ruff
- **Type checker**: Pyright (strict mode)

All code must have full type annotations. Never use `Any` or `object`:

```python
# Good
def process_items(items: list[Item], *, threshold: float = 0.5) -> list[Item]:
    ...

# Bad: missing annotations
def process_items(items, threshold=0.5):
    ...
```

Use modern Python syntax:

```python
# Good
def get_item(id: UUID) -> Item | None: ...
items: list[str] = []

# Bad: old typing syntax
from typing import Optional, List
def get_item(id: UUID) -> Optional[Item]: ...
items: List[str] = []
```

### Docstrings

Use NumPy-style docstrings for all public functions, classes, and methods:

```python
def construct_list(
    items: list[Item],
    *,
    constraints: list[Constraint],
    max_size: int = 100,
) -> ExperimentList | None:
    """Construct an experiment list from items with constraints.

    Parameters
    ----------
    items
        Items to partition into the list.
    constraints
        Constraints the list must satisfy.
    max_size
        Maximum items in the resulting list.

    Returns
    -------
    ExperimentList | None
        A balanced list, or None if constraints cannot be satisfied.

    Raises
    ------
    ValueError
        If max_size is less than 1.

    Examples
    --------
    >>> items = [Item(...), Item(...)]
    >>> constraints = [UniquenessConstraint(field="lemma")]
    >>> result = construct_list(items, constraints=constraints)
    """
```

### Commit Messages

- One sentence, present tense, ends with a period
- Describe what the change does, not what you did

```
# Good
Adds list constraint solver with backtracking.
Fixes type error in item template validation.
Updates documentation for ListCollection API.

# Bad
Add list constraint solver          # imperative, no period
Added list constraint solver.       # past tense
fix: constraint solver              # not a sentence
Fixed the bug                       # too vague
```

### Writing Style

- Never use em-dashes in code, documentation, or comments
- Avoid generic filler phrases ("This powerful module elegantly...")
- Be direct and specific

## Recognition

Contributors are recognized in the project's release notes. Thank you for helping make bead better!

---

Questions? Open a [discussion](https://github.com/FACTSlab/bead/discussions) or reach out to the maintainers.
