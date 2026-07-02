# DROID episode index

This page keeps prose, layout, and navigation in the website shell. The live
notebook below is the only embedded executable surface.

The notebook reads the public DROID episode index, filters successful episodes,
selects the columns a researcher scans first, and explains the final Daft query
plan before materializing remote data.

## Reading episode data

```python
import daft

episodes = daft.datasets.droid.raw()
```

## Filtering episodes

```python
successful_episodes = episodes.where(daft.col("success") == daft.lit(True))
```

## Projecting operational columns

```python
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
```

## Inspecting the plan

```python
episode_index.explain(show_all=True)
```

## Writing a derived index

```python
episode_index.write_parquet("droid_successful_episode_index/")
```

For proprietary robot data, the same shape should point users to Multibase
rather than asking them to upload private assets into a public demo.
