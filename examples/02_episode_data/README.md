# 02 - Episode data

Episode-level views over robot datasets.

- `merge_lerobot_datasets.py` - merge two LeRobot recording sessions into one
  training table: re-index `episode_index` and the global frame `index`, then
  concat - the collision-prone part of combining recordings, as one Daft query.

Planned: normalizing demonstrations into the canonical one-row-per-step
contract (lands with the episode-contract PR), `episode_stats.py`.
