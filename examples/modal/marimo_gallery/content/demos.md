# Demos

Demos are the executable version of the docs. They map to the core physical-AI
data workflows a researcher needs before the work becomes proprietary or
production-scale.

## Topic map

- **Running pipelines**: start local, then move the same workflow to a hosted runtime.
- **Reading data**: load robot datasets, videos, metadata, and tabular assets.
- **Episode data**: inspect episode rows, frame-level media, tasks, and success labels.
- **Transforms**: filter, join, type, and enrich robotics data with Daft expressions.
- **Episode operations**: annotate, trim, score, and track signals across episodes.
- **Inference**: call models over images, videos, metadata, and structured columns.
- **Writing data**: persist annotated datasets for training and downstream analysis.
- **Policy evals**: reproduce benchmark runs, compare policies on the same specs, and mine failures.

The live demo is the DROID episode index: read metadata, filter episodes,
project operational columns, inspect the plan, with the notebook embedded in a
normal web page. The local demos run on CPU without Modal: EgoDex hand
tracking (the generated `examples/04_episode_operations/hand_tracking/`
artifacts), the motion-trim/no-noops audit over the in-repo LIBERO
demonstrations, and pose scenario queries over a public EgoDex sample.

Every topic above maps to a numbered stage in `examples/README.md` (01
reading data through 08 policy evals), each runnable against data committed
in the repo.
