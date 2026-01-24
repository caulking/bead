# Contributing to bead

Thank you for your interest in contributing to bead. This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

## Code of Conduct

This project follows a standard code of conduct. Be respectful, inclusive, and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) for package management
- Git

### Setting Up the Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/bead.git
cd bead

# Install all dependencies
uv sync --all-extras

# Verify the setup
uv run pytest tests/ -x
```

**Important:** Always use `uv run` to execute commands. Do not activate the virtual environment manually.

### Useful Commands

```bash
# Run tests
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=bead --cov-report=html

# Lint code
uv run ruff check .

# Format code
uv run ruff format .

# Type check
uv run pyright

# Run all checks
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

## Development Workflow

1. **Create a branch** from `main` for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines below.

3. **Write or update tests** for your changes.

4. **Run all checks** before committing:
   ```bash
   uv run ruff check .
   uv run ruff format .
   uv run pyright
   uv run pytest tests/
   ```

5. **Commit your changes** following the commit message convention.

6. **Push and create a pull request.**

### Commit Message Convention

All commit messages must follow these rules:

- Exactly one sentence
- Present tense (not imperative)
- Ends with a period
- Never mention Claude, AI, or assistant

**Good examples:**
```
Adds list constraint solver with backtracking.
Fixes type error in item template validation.
Updates Python version requirement to 3.13+.
```

**Bad examples:**
```
Add list constraint solver          # imperative mood, no period
Added list constraint solver.       # past tense
fix: constraint solver              # not a sentence
```

## Pull Request Process

1. **Ensure all checks pass.** PRs with failing tests or linting errors will not be reviewed.

2. **Update documentation** if you change any public APIs.

3. **Add tests** for new functionality. Target >90% coverage for new code.

4. **Keep PRs focused.** One feature or fix per PR. Large PRs are harder to review.

5. **Write a clear description** explaining what the PR does and why.

6. **Link related issues** using keywords like "Fixes #123" or "Closes #456".

### PR Checklist

- [ ] All tests pass (`uv run pytest tests/`)
- [ ] Code is formatted (`uv run ruff format .`)
- [ ] No linting errors (`uv run ruff check .`)
- [ ] Type checking passes (`uv run pyright`)
- [ ] Documentation updated if needed
- [ ] Commit messages follow convention

## Code Style

### Python

- **Python version:** 3.13+ with modern syntax
- **Line length:** 88 characters
- **Formatting:** Ruff
- **Linting:** Ruff
- **Type checking:** Pyright in strict mode

### Type Annotations

All code must have full type annotations:

```python
# Good
def process_items(items: list[Item], *, threshold: float = 0.5) -> list[Item]:
    ...

# Bad (missing annotations)
def process_items(items, threshold=0.5):
    ...
```

**Never use `Any` or `object`** as type annotations. Use specific types or type aliases:

```python
# Good
type MetadataValue = str | int | float | bool | None | dict[str, MetadataValue] | list[MetadataValue]

def load_config(path: Path) -> dict[str, MetadataValue]:
    ...

# Bad
def load_config(path: Path) -> dict[str, Any]:
    ...
```

### Modern Python Syntax

Use modern Python 3.13+ syntax:

```python
# Good: Union with |
def get_item(id: UUID) -> Item | None:
    ...

# Bad: Optional from typing
from typing import Optional
def get_item(id: UUID) -> Optional[Item]:
    ...

# Good: Built-in generics
def process(items: list[str]) -> dict[str, int]:
    ...

# Bad: Generics from typing
from typing import List, Dict
def process(items: List[str]) -> Dict[str, int]:
    ...
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

    Partitions the given items into a balanced list that satisfies
    all specified constraints.

    Parameters
    ----------
    items
        List of experimental items to partition.
    constraints
        List of constraints the list must satisfy.
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

### Writing Style

**Never use em-dashes** in code, documentation, or comments. Use:
- Commas for mild separation
- Semicolons for related clauses
- Colons for elaboration
- Parentheses for asides

**Avoid AI slop** (generic, hollow, or formulaic writing):
- Bad: "This powerful module elegantly handles..."
- Good: "This module handles list partitioning by..."

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage report
uv run pytest tests/ --cov=bead --cov-report=html

# Run specific test file
uv run pytest tests/items/test_item.py -v

# Run tests matching a pattern
uv run pytest tests/ -k "test_construct"
```

### Writing Tests

- Place tests in the `tests/` directory mirroring the source structure
- Use pytest fixtures for shared test data
- Use descriptive test names: `test_<action>_<condition>`
- Include docstrings explaining what each test verifies

```python
import pytest
from bead.items import Item, ItemTemplate


@pytest.fixture
def sample_items() -> list[Item]:
    """Create sample items for testing."""
    return [...]


class TestItemTemplate:
    """Tests for ItemTemplate class."""

    def test_constructs_items_from_templates(
        self, sample_items: list[Item]
    ) -> None:
        """Should construct items from filled templates."""
        template = ItemTemplate(judgment_type="acceptability")
        items = template.construct_items(sample_items)

        assert len(items) > 0
        assert all(i.judgment_type == "acceptability" for i in items)

    def test_returns_empty_list_for_empty_input(self) -> None:
        """Should return empty list when given no templates."""
        template = ItemTemplate()
        items = template.construct_items([])

        assert items == []
```

### Test Coverage

- Target >90% coverage for new code
- Focus on testing behavior, not implementation details
- Include edge cases (empty input, None values, invalid data)

## Documentation

### Updating Documentation

- Update docstrings when changing public APIs
- Add examples for new functionality
- Keep the README current with new features

### Building Documentation

```bash
# Build and serve documentation locally
uv run mkdocs serve

# Build static documentation
uv run mkdocs build
```

## Reporting Bugs

When reporting bugs, please include:

1. **Description:** Clear, concise description of the bug
2. **Steps to reproduce:** Minimal code example that demonstrates the issue
3. **Expected behavior:** What you expected to happen
4. **Actual behavior:** What actually happened
5. **Environment:** Python version, OS, bead version
6. **Error messages:** Full traceback if applicable

Use this template:

```markdown
## Description
Brief description of the bug.

## Steps to Reproduce
```python
from bead import ...

# Minimal code that reproduces the issue
```

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- Python version: 3.13.x
- OS: macOS/Linux/Windows
- bead version: 0.1.0

## Error Message
```
Full traceback here
```
```

## Suggesting Features

When suggesting features, please include:

1. **Use case:** Why is this feature needed? What problem does it solve?
2. **Proposed solution:** How do you envision the feature working?
3. **Alternatives:** Have you considered other approaches?
4. **Examples:** Code examples showing how the feature would be used

## Questions?

If you have questions about contributing, open a discussion or reach out to the maintainers.

Thank you for contributing to bead!
