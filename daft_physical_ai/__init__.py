"""Physical-AI data annotation, episode analysis, and policy evals on Daft.

`track_hands(images, method=...)` takes an image column (a Daft expression) and
returns a hand-pose column, so it composes with any Daft pipeline. Every method
returns the same output schema - see `HANDS_DTYPE`.

`Episode` and `Step` provide a canonical one-row-per-step table contract for
robot episodes and rollouts, so datasets and evaluations land in one
Daft-readable parquet layout.

`daft_physical_ai.evals` is the analysis half of the eval loop over that
contract: success rates, per-spec policy comparison, failure labeling, and
benchmark-protocol validation.
"""

from __future__ import annotations

from .episodes import Episode, ROLLOUT_SCHEMA, Step
from .hands import HAND_DTYPE, HANDS_DTYPE

__all__ = ["HANDS_DTYPE", "HAND_DTYPE", "ROLLOUT_SCHEMA", "Episode", "Step"]
