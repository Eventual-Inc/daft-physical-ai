# 02 - Episode data

Episode-level views over the canonical step rows: normalize a dataset into
`daft_physical_ai.episodes` step rows, then inspect it.

- `normalize_libero_demos.py` - download the original LIBERO-Spatial
  demonstration HDF5 from Hugging Face and normalize it through
  `daft_physical_ai.ingest.Hdf5Ingestor` (one canonical parquet part per
  demo, signals only). The full suite - 500 demos, 62,250 step rows,
  ~4.7 MB - is committed under [`data/`](data/README.md), so downstream
  stages run offline.

Planned: `episode_stats.py` (durations, frame counts, success rates,
gripper-transition counts), `merge_lerobot_datasets.py` (needs the next
Daft release).
