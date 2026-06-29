# Resources

- https://docs.daft.ai for the user-facing API docs for Daft
- https://docs.daft.ai/en/stable/extensions/authoring/ for writing Daft extensions
- - https://docs.daft.ai/en/stable/api/udf/ for `@daft.func`, `@daft.cls`, and `@daft.udaf`


# Dev Workflow

1. Set up Python environment, install dependencies, and build dev package: `uv sync`
2. Activate .venv: `source .venv/bin/activate`
3. Run tests: `uv run pytest tests/ -v`
4. To use the LeRobot reader (`daft.datasets.lerobot`), install a nightly Daft -
   it is merged ([Daft #7090](https://github.com/Eventual-Inc/Daft/pull/7090))
   but not yet in a released version (latest is v0.7.16):
   `uv pip install --prerelease=allow --extra-index-url https://nightly.daft.ai -U daft`

# TODO

- [ ] Switch off the Daft nightly once `daft.datasets.lerobot` ships in a release
  (> v0.7.16): bump the `daft` floor in `pyproject.toml`, drop the nightly
  install step above, and re-run `uv lock`.

# PR Conventions

- Titles: Conventional Commits format; enforced by `.github/workflows/pr-labeller.yml`.
- Descriptions: follow `.github/pull_request_template.md`.
