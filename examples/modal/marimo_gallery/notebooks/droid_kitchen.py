from __future__ import annotations

import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    from contextlib import redirect_stdout

    import daft
    import marimo as mo

    return daft, io, mo, redirect_stdout


@app.cell
def _(mo):
    mo.md(
        """
        # DROID episode index

        This notebook runs as a Marimo app on Modal. Daft executes in the
        backend Python container; the browser is only the interactive UI.

        The workflow reads the public DROID episode index, filters successful
        episodes, selects the operational columns a researcher would inspect
        first, and explains the lazy Daft plan before materializing data.
        """
    )


@app.cell
def _(daft):
    # Read the public raw DROID episode index.
    episodes = daft.datasets.droid.raw()

    # Filter to successful episodes. Keep this lazy; no remote data is read yet.
    successful_episodes = episodes.where(daft.col("success") == daft.lit(True))

    # Project the columns that make the episode index useful for scanning.
    episode_index = successful_episodes.select(
        "uuid",
        "scene_id",
        "building",
        "current_task",
        "success",
        "trajectory_length",
        "wrist_video",
        "ext1_video",
        "ext2_video",
    )

    return episode_index, episodes, successful_episodes


@app.cell
def _(episode_index, mo):
    mo.md("## Query preview")
    episode_index.limit(10)


@app.cell
def _(episode_index, io, mo, redirect_stdout):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        episode_index.explain(show_all=True)

    mo.md(
        f"""
        ## Daft query plan

        ```text
        {buffer.getvalue().strip()}
        ```
        """
    )


if __name__ == "__main__":
    app.run()
