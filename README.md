# daft-physical-ai

Physical-AI data annotation on [Daft](https://docs.daft.ai), starting with hand
tracking. The annotation methods run as Daft UDFs, so they slot into any Daft
pipeline and execute lazily, batched, and distributed.

> **Status:** early scaffold. The API below is the planned design (see
> [design doc](https://github.com/Eventual-Inc/multibase/pull/527)); the
> `hands/` implementation is not built yet.

## Planned API

The package operates on an image column and returns a hand-pose column - it
doesn't know about LeRobot, EgoDex, or Modal, so it composes with any Daft
DataFrame:

```python
import daft
from daft_physical_ai.hands import track_hands

df = daft.read_parquet("frames.parquet")   # any df with an image column

# wilor -> GPU, 3D MANO keypoints, both hands (MANO weights user-supplied)
df = df.with_column("hands", track_hands(df["image"], method="wilor", mano_path="MANO_RIGHT.pkl"))

# mediapipe -> CPU, 2D only, permissive license, no weights to supply
df = df.with_column("hands", track_hands(df["image"], method="mediapipe"))

df.write_parquet("annotated/")
```

One unified output schema regardless of method
(`kp3d` is null for MediaPipe):
`list[struct{ handedness, confidence, kp2d, kp3d? }]`.

## Development

```bash
uv sync                      # set up env + install deps
uv run pre-commit install    # install lint/format hooks
uv run pytest tests/ -v      # run the test suite
```

## Versioning

Versions are derived from git tags via `hatch-vcs`. Tag releases as `v0.1.0`,
`v0.2.0`, etc.

## Publishing

Publishing a GitHub release triggers `.github/workflows/publish-package.yml`,
which builds a wheel and sdist with `uv build` and uploads both to PyPI via
[trusted publishing](https://docs.pypi.org/trusted-publishers/). Configure the
trusted publisher on PyPI for this repository before the first release.
