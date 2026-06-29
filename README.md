# daft-ext-py-template

A template for creating a Python-only Daft extension. If you're looking to create a Python + Rust Daft extension, see the [daft-ext-rust-template](https://github.com/Eventual-Inc/daft-ext-rust-template) instead.

## Usage

1. Clone this repository: `git clone https://github.com/Eventual-Inc/daft-ext-python-template.git`
2. Rename the `daft_ext_template/` package directory and update `name` in `pyproject.toml`
3. Fill in the remaining blanks in `pyproject.toml` (description, authors, repository URL, etc.)
4. Install dependencies: `uv sync`
5. Install pre-commit hooks: `uv run pre-commit install`
6. Run the example test suite: `uv run pytest tests/ -v`
7. Start developing!

## Example

This template ships a minimal `greet` function in `daft_ext_template/__init__.py`:

```python
import daft
from daft_ext_template import greet

df = daft.from_pydict({"name": ["Ada", "Grace"]})
df.select(greet(df["name"]).alias("greet")).show()
```

See the [Extensions overview](https://docs.daft.ai/en/stable/extensions/overview/) and [UDF API docs](https://docs.daft.ai/en/stable/api/udf/) for patterns like batch UDFs, class UDFs, and custom aggregations.

## Versioning

Versions are derived from git tags via `hatch-vcs`. Tag releases as `v0.1.0`, `v0.2.0`, etc.

## Publishing

Publishing a GitHub release triggers `.github/workflows/publish-package.yml`, which builds a wheel and sdist with `uv build` and uploads both to PyPI via [trusted publishing](https://docs.pypi.org/trusted-publishers/). Configure the trusted publisher on PyPI for this repository before your first release.
