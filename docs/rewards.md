# Reward scoring

Score episodes with a reward model
([Robometer-4B](https://huggingface.co/robometer/Robometer-4B)) - per-frame
task progress (0-1) plus success probability, written back as a dataset column.
Use it to filter failed or stalled episodes before BC training, as dense reward
for RL post-training, or to catch mislabeled tasks.

```python
from daft_physical_ai.rewards import score_rewards

# one row per episode: task text, length, and where its frames live in the video
# (e.g. from daft.datasets.lerobot.read_episodes - the video column can be a
# Daft file handle or a local path string)
df = df.with_column(
    "rewards",
    score_rewards(
        df["task"], df["length"], df["from_ts"], df["to_ts"], df["video"],
        url="http://localhost:8001",   # any running Robometer eval server
        max_frames=8,                  # frames sampled per episode
    ),
)
```

Scoring is a pure HTTP call: the package never imports the model - you bring a
running [Robometer eval server](https://github.com/robometer/robometer) and
pass its URL. `daft-physical-ai rewards` scaffolds a complete demo plus the two
server scripts to run one yourself (`run_robometer_server.py` for any NVIDIA
GPU, `modal_eval_server.py` for [Modal](https://modal.com)). The output type:

```
struct {
    reward_score:       list[float64]                          # per-frame task progress, 0-1
    robometer_success:  list[float64]                          # per-frame success probability
    reward_frames:      list[struct{index, timestamp_s}]       # which frames were scored
}
```

## Example

[examples/rewards/](../examples/rewards/) is the executed walkthrough - read
LIBERO episode metadata, score each episode with `score_rewards`, plot the
progress curves, and filter low-progress episodes with a Daft query. The
Robometer server scripts it talks to are committed next to it.

Generate your own (different dataset, episode count, frame budget):

```bash
daft-physical-ai rewards    # interactive
daft-physical-ai rewards --episodes 10 --max-frames 8 --no-input
```
