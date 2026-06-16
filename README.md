---
title: "cibica"
author: "Torbjörn E. M. Nordling"
date: "2026-06-16"
license: "Apache-2.0"
purpose: "Python package with src layout and packaging configuration."
---

# cibica

## Description

Circle estimation by CIBICA with example datasets.

---

## Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) (>= 0.7) for Python environment and package management.

**Install uv:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you do not have Python installed, uv can install it for you:

```bash
uv python install 3.11
```

## Installation

There are two ways to install: **globally on `PATH`** for everyday use (`uv tool install`), or **locally in the project** for development (`uv pip install`).
Adding the `-e` flag installs in editable mode, linking back to the source tree so code changes take effect immediately without reinstalling.
Adding `.` tells uv to read `pyproject.toml` from the current directory for the package name, version, dependencies, and entry points, alternatively add the path.

### Global CLI tool (`~/.local/bin/`)

`uv tool install` creates an isolated virtual environment (in `~/.local/share/uv/tools/`) and symlinks entry-point scripts into `~/.local/bin/`:

| Source | Command |
|--------|---------|
| PyPI (when published) | `uv tool install cibica` |
| Git | `uv tool install git+https://github.com/nordlinglab/cibica` |
| Local folder | `cd cibica && uv tool install -e .` |

### Local in project (`.venv/bin/`)

`uv pip install` installs the package into the project virtual environment:

| Source | Command |
|--------|---------|
| PyPI (when published) | `uv pip install cibica` |
| Git | `uv pip install git+https://github.com/nordlinglab/cibica` |
| Local folder | `cd cibica && uv pip install -e .` |

For editable local installs for development, first create and activate the virtual environment as shown in [Development](#development).

### How uv selects the target environment

`uv pip install` places packages into the first environment it finds:

1. Active virtual environment (`$VIRTUAL_ENV`)
2. `.venv/` in the current or parent directories
3. System Python (if no virtual environment is found)

This is why editable installs land in `.venv/bin/` rather than `~/.local/bin/` — the project's `.venv/` is detected automatically.

---

## Usage

```python
from cibica import example_function

result = example_function()
```

---

## Development

### Setup Development Environment

```bash
git clone https://github.com/nordlinglab/cibica
cd cibica
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```
Note:
- `ninit` already creates `.venv/` during scaffolding, so you can skip `uv venv .venv` in a freshly created project.
- `source .venv/bin/activate` sets the `VIRTUAL_ENV` environment variable and prepends `.venv/bin/` to `PATH` in your **current shell session only**.
  This makes `python`, `pytest`, and other commands resolve to the project-local virtual environment instead of the system or user-level Python.
  It does **not** modify any global state, other shell sessions, or persist after the shell exits.
  Activation is needed when running tools that rely on `PATH` lookup (e.g. `pytest`, `mypy`, `black`).
  It is **not** needed for `uv run` or `uv pip` commands, which locate the virtual environment automatically.
- `[dev]` installs the optional dependency group named dev declared in `pyproject.toml`.

### Running Tests

```bash
pytest
pytest --cov=cibica  # with coverage
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/
# Or with flake8:
flake8 src/ tests/

# Type checking
mypy src/
```

---

## API Documentation

### Option 1: Sphinx

```bash
# Install sphinx
uv pip install sphinx sphinx-rtd-theme

# Generate docs
cd doc
sphinx-quickstart
make html
```

### Option 2: MkDocs

```bash
# Install mkdocs
uv pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve docs locally
mkdocs serve

# Build docs
mkdocs build
```

### Docstring Style

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Short description of function.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: If param1 is empty.

    Example:
        >>> example_function("hello", 42)
        True
    """
    pass
```

---

## VSCode Setup

### Recommended Extensions

1. **Python** (Microsoft)
2. **Pylance** (Microsoft) - Fast IntelliSense
3. **Black Formatter** (Microsoft)
4. **Ruff** (Charlie Marsh)

### settings.json

Add to your VSCode workspace settings:

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  },
  "python.analysis.typeCheckingMode": "basic",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"]
}
```

---

## Style Guide

### PEP 8 Compliance

This project follows [PEP 8](https://peps.python.org/pep-0008/) with these tools:

| Tool | Purpose | Config |
|------|---------|--------|
| **Black** | Code formatting | `pyproject.toml` |
| **Ruff** | Linting (fast) | `pyproject.toml` |
| **mypy** | Type checking | `pyproject.toml` |
| **isort** | Import sorting | handled by Ruff |

### Key Conventions

- Line length: 88 characters (Black default)
- Imports: absolute, sorted alphabetically
- Naming: `snake_case` for functions/variables, `PascalCase` for classes
- Type hints: required for public APIs

---

## Project Structure

```
cibica/
├── src/
│   └── cibica/
│       ├── __init__.py
│       └── ...
├── tests/
│   ├── __init__.py
│   └── test_*.py
├── doc/
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Contributing

### Pull Request Process

1. **Create a branch** (never commit directly to main/master):
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes and commit** (use conventional commits):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `test:` adding tests
   - `refactor:` code restructuring

3. **Run checks before pushing**:
   ```bash
   black src/ tests/
   ruff check src/ tests/
   mypy src/
   pytest
   ```

4. **Push and create PR**:
   - GitHub: `gh pr create --title "feat: description"`
   - Bitbucket: Create via web UI
   - GitLab: `glab mr create`

### Issue Classification

- `bug` - Something isn't working
- `enhancement` - New feature request
- `documentation` - Documentation improvements
- `question` - Questions about usage

---

## References

<!-- Use Pandoc citation syntax in your text: [@citekey], [@citekey, p. 42], or @citekey.
Each reference must include a DOI, ISBN, or URL for verification. Run `nref <file>.md` to validate and generate .bib file. Format: APA7 (American Psychological Association, 7th edition). -->

Van Rossum, G., Warsaw, B., & Coghlan, A. (2001). PEP 8 - Style Guide for Python Code.
    https://peps.python.org/pep-0008/

Python Packaging Authority. (2024). Python Packaging User Guide.
    https://packaging.python.org/
