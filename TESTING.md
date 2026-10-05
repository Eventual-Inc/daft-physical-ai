# Testing

Two layers: the committed unit suite (CI, CPU-only) and out-of-band real-data
runs whose results are recorded here.

## Unit suite (CI)

```bash
uv sync && uv run pytest tests/ -v
```

- MediaPipe facade + a real run asserting the output dtype equals `HANDS_DTYPE`.
  The real run skips without the `[mediapipe]` extra.
- WiLoR facade: `mano_path` required; the expression's dtype equals `HANDS_DTYPE`
  (built without executing - `@daft.cls` is lazy, no GPU needed); `ensure_assets`
  places a provided MANO file. CLI: all method x runtime combos render to valid
  Python + `.ipynb`.

CI runs on Python 3.10 + 3.13; ruff + mypy via pre-commit.

## Real-data runs

**MediaPipe (CPU, 2D)** - verified locally on `pepijn223/egodex-test` via
`daft.datasets.lerobot`: 10 frames -> 9 hands. Byte-identical to the original
reference implementation (`max_kp2d_abs_diff=0.0` over 24 hands).

**WiLoR (GPU, 3D)** - needs CUDA, so verified on Modal (an L4); the package has no
Modal dependency and runs on any CUDA GPU. The env needs a CUDA `torch` build, the
`[wilor]` extra plus `chumpy` from git
(`pip install 'chumpy @ git+https://github.com/mattloper/chumpy'`), and
`ensure_assets()` to fetch the repo + weights and place `MANO_RIGHT.pkl`. Then
`track_hands(images, method="wilor", mano_path=..., wilor_root=...)`: 12 frames ->
24 hands with real 3D `kp3d`, byte-identical to the reference
(`max_kp2d_diff=max_kp3d_diff=0.0`).

> **torch import order:** WiLoR segfaults if `torch`/CUDA is first imported inside
> the `@daft.cls` worker, so `track_hands(method="wilor")` imports it in the
> caller's process first. A Daft+torch interaction, not Modal- or
> concurrency-specific.

**ABC-130k (gated Hugging Face, MCAP)** - the unit suite covers every
`daft_physical_ai.datasets.abc` function against synthetic MCAPs, including
real H.264/H.265 streams encoded with PyAV. The two real-data tests are marked
`integration` and skip without `HF_TOKEN`; they pin
`XDOF/ABC-130k@29136bc9` episode `5b33995f-ba4a-49f8-bfb7-c6c034df0865`
(`clip_the_socks_to_the_hanger`, 9,849 messages, 18 chunks, 9 annotations):

```bash
HF_TOKEN=hf_... uv run --extra abc pytest tests/test_datasets_abc.py -v -m integration
HF_TOKEN=hf_... uv run --extra abc python examples/abc_episode_messages.py \
  --task clip_the_socks_to_the_hanger --frames
```

Not yet run from this package: the expected counts come from the earlier
Daft#7248 smoke run of the same pinned episode.

**REASSEMBLE (TU Wien HDF5 + Hugging Face LeRobot port)** - the unit suite
covers every `daft_physical_ai.datasets.reassemble` function against synthetic
HDF5 files laid out like the release: MP4 blobs as `|V` scalars, MP3 bytes as
int64 arrays, a recording with no `hand` camera. It also covers a local
stand-in for the port's side folders and a served zip for `download()`. Two
real-data tests are marked `integration`:

- `test_reassemble_tuwien_smoke` runs when `REASSEMBLE_ROOT` is set. It
  extracts `2025-01-11-14-43-37` there with `download()` if it is missing, then
  checks 4 actions, 2 skills (Grasp ok, Lift failed), `measured_force`
  (23,240, 3), audio at 16/16/48 kHz with 372,096/372,096/1,123,200 samples,
  2,551,295 events, and 642 hand + 136 event-camera frames.
- `test_reassemble_lerobot_port_sidecars_smoke` runs when `HF_TOKEN` is set.
  It pins `robot-lev/reassemble@37d242d3` and checks 149 episodes
  (111 train / 37 test / 1 unassigned), episode 21 hand audio recovered at
  48 kHz with 1,123,200 samples, and 2,551,295 events.

```bash
REASSEMBLE_ROOT=/data/reassemble HF_TOKEN=hf_... \
  uv run pytest tests/test_datasets_reassemble.py -v -m integration
HF_TOKEN=hf_... uv run python examples/reassemble_contact_segments.py /data/reassemble --compare-port
```

**Real-data run (2026-10-04, macOS arm64, daft 0.7.20):** both integration
tests passed in 12 s from an empty `REASSEMBLE_ROOT`, which included the live
24 MB extraction. The example ran end to end from an empty directory: 4
actions, peak force 28.5 N during "Pick square peg 3." (14,381 F/T samples),
2,020,836 events in that 14.4 s window, and hand frames 60+ at 224x168. With
`--compare-port` it reported "LeRobot port episode 21: 696 frames at 30 fps; the
HDF5 holds 23240 force/torque samples (33x more)". A wider manual check on
`2025-01-10-15-39-56` and `2025-01-10-16-17-40` (no hand camera: 0 hand rows,
387 event-camera frames) also passed.

## CLI scaffolder (`daft-physical-ai`)

**Unit tests** (`tests/test_cli.py`, in CI): all 6 method x runtime combos render
a `demo.py` that `compile()`s and a `demo.ipynb` whose every code cell `compile()`s;
the Modal script uses `@app.local_entrypoint()` while the Modal notebook uses
`with app.run():` (no entrypoint); config validation (wilor needs `mano_path`; bad
method/runtime/limit); CLI exit codes (0 ok / 1 file exists / 2 bad args) and
`--force` overwrite.

**Runtime policy:** MediaPipe is CPU-only, so it's always `local` - the CLI skips
the runtime question for it and overrides `--runtime modal` to local. `modal` is
only offered when WiLoR (GPU) is involved.

**Interactive prompts** (driven via tmux): default is interactive; flags pre-fill
answers and an explicitly-passed flag is not re-prompted; `--no-input`/non-tty runs
fully non-interactively. Verified: method/runtime/dataset/image prompts, the MANO
prompt appearing only for wilor/both, the runtime prompt skipped for mediapipe,
invalid-choice re-prompt, `--dataset` suppressing its own prompt, and the Modal
login reminder (`modal setup`) printing for the modal runtime.

**Generated demos executed:**

- **Local MediaPipe** - both `python demo.py` and the notebook (headless via
  `nbconvert --execute`, all cells run, the final `.show()` renders Daft's HTML
  table).
- **With `--with-eval`** - the demo's EgoDex scoring ran end to end on CPU and
  produced sensible metrics (e.g. 12 frames: `detect=100% mean_err=0.105
  PCK@.1/.2/.3 = 54/88/97`). The committed `examples/hands/demo.{py,ipynb,md}` are this
  MediaPipe+eval demo, regenerated by `scripts/regen_demo.py` (see "Regenerating
  the examples demo" in `AGENTS.md`): it executes the notebook (`DAFT_PROGRESS_BAR=0`
  so progress bars stay out of the outputs) and derives the markdown's outputs +
  the separate `demo_keypoints.png` from that one executed copy, so the formats
  can't drift.
- **Three mediums stay equivalent** - notebook and markdown render from one shared
  cell list; a test asserts every notebook code cell appears verbatim in the
  markdown.
- **Modal (MediaPipe pipeline)** - a generated Modal script run end to end with
  `modal run`: image builds, dataset downloads, inference runs on Modal, and 6
  annotated frames return (`got 6 frames back from Modal`). This caught a real bug
  - the image was missing `libGLESv2.so.2` (now `libgles2`/`libegl1` are in the
  apt list) - and exercises the same Modal image + pipeline the `both`/WiLoR modal
  demos use.

**Published-package (PyPI) end to end** - verified on v0.1.2 (2026-07-09), all
from PyPI installs, no repo code:

- **Fresh local install** - `pip install "daft-physical-ai[mediapipe]"` into a
  clean Python 3.13 venv, CLI scaffolds a demo, the demo's pipeline (remote
  `lerobot.read` + decode + `track_hands` + `.show()`) runs in ~5s warm with real
  hand detections. This flow is what caught the two v0.1.1 packaging bugs fixed
  in v0.1.2 (missing av/pillow, now via `daft[video]`; a stale
  `python_version < '3.13'` marker silently skipping mediapipe).
- **Modal WiLoR script** - generated with `--method wilor --runtime modal`, run
  with `modal run`: image builds from PyPI, WiLoR assets fetched, L4 inference,
  `annotated 12 frames` / `got 12 frames back from Modal`.
- **Modal WiLoR notebook** - the same demo's `.ipynb` executed headless
  (`nbconvert --execute`) on a Python 3.11 kernel: the notebook's
  `with app.run():` drives the full Modal roundtrip, same 12 annotated frames.

**Known limitations (not bugs):**

- Generated demos `pip install daft-physical-ai`, so they need the package
  published to PyPI. Local-runtime demos just need it importable locally.
- The Modal **notebook** path (`app.run()`) serializes the notebook-defined
  function, so the kernel's Python must match the image (3.11) - verified
  working on a 3.11 kernel above; other kernel versions fail serialization.

## Reward scoring (`rewards/` + `daft-physical-ai rewards`)

**Unit tests** (`tests/test_rewards.py`, `tests/test_cli_rewards.py`, in CI):
frame sampling asserted identical to Macrodata refiner's `_sample_indexes`
(endpoints included, `max_frames` respected); request build/parse round-trips;
`score_rewards` end to end against a synthetic mp4 + a fake local Robometer
HTTP server (output dtype equals `REWARD_DTYPE`, sampled indexes and
timestamps line up with the video's frame clock, `max_frames` reaches the
UDF). CLI: rendered script/notebook `compile()`, markdown and notebook share
content, the two server scripts land verbatim from the package templates,
exit codes (0/1/2) and `--force`.

**Real-data runs (2026-07-14, LIBERO `nvidia/LIBERO_LeRobot_v3`):**

- **Modal serving from a fresh scaffold** - `daft-physical-ai rewards
  --no-input` into an empty dir, then `modal deploy modal_eval_server.py`
  straight from it: full image build (robometer install + model bake at the
  pinned commit/revision) and deploy in ~11 min, health endpoint 200 behind
  proxy auth.
- **Generated script against Modal** - the scaffolded `demo.py` run as its
  post-scaffold hint suggests (`uv run --with daft-physical-ai --with
  huggingface_hub --with matplotlib`, `ROBOMETER_URL` + `MODAL_KEY`/
  `MODAL_SECRET`): 3 episodes scored through the proxy-auth endpoint,
  progress/success values consistent with the multibase-validated baseline
  (e.g. ep0 climbs 0.10 -> 0.92, final success 0.97).
- **Committed example** - `examples/rewards/` regenerated headless by
  `ROBOMETER_URL=... python scripts/regen_demo.py --demo rewards`
  (`nbconvert --execute`, 5 episodes): all cells run, the progress-curve
  figure renders to `demo_progress.png`, and the closing Daft filter flags
  ep1 + ep3 by final-frame success < 0.5.
- **Numeric diff against the multibase baseline (2026-07-15)** - the committed
  example's 5 episodes compared per-frame against the post-2 validated
  results (multibase `GTM/content/post-2-open/open-results.json`): keep/drop
  decisions identical on all 5, final success within 0.09, per-frame progress
  within 0.02 on 3 of 5 (worst 0.125 on ep2) - cross-deployment model noise,
  same band as the original 25-episode validation.
- **Re-run after the guard removal (2026-07-15)** - post-rebase onto main
  (daft >= 0.7.20, Jupyter guard dropped): fresh `modal deploy` from the
  example's server script, the committed example regenerated against it
  (scores identical to the 07-14 run), and a fresh scaffold's `demo.py` run
  end to end (3 episodes, values identical, the closing filter flags ep1).
- **read_episodes rewrite (2026-07-16)** - the demo rebuilt on
  `daft.datasets.lerobot.read_episodes` (episode rows + video file handles
  streamed from the Hub; no `hf_hub_download`, no hardcoded chunk paths) and
  `score_rewards` extended to accept a Daft file handle: fresh `modal deploy`,
  the committed example regenerated against it - all 5 episodes' per-frame
  progress and success identical to the 07-15 run, closing filter still flags
  ep1 + ep3.

**Known limitation (not a bug):** executing the rewards demo or its regen
needs a live Robometer eval server (`ROBOMETER_URL`); CI exercises the
client against the fake in-process server only.
