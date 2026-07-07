# Modal deployment recipes

Hosted runtimes for the examples - never a required dependency. Model code
stays Modal-free in the package; these apps own only images, volumes, and
entrypoints.

- [`marimo_gallery/`](marimo_gallery/) - FastAPI docs/gallery site with an
  embedded Marimo notebook running Daft, deployable with `modal serve`.
