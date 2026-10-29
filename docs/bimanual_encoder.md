# Joint bimanual YAM motion encoder

This experiment replaces the two motion encoders with one encoder over
`[left_joint1..6, right_joint1..6]`. The left and right posture GMMs remain
separate, and their existing artifacts are reused unchanged. The default
marginal training and original demo are still available.

## Training

```sh
uv sync --extra training --extra planning --extra simulation
uv run --extra training python scripts/train_yam.py --bimanual --gpu 3
```

This reuses `work/yam/corpus`: exactly the same manifest, 15,000 training episodes
and 600 validation episodes per source, 50/50 ABC/MolmoAct2 minibatches, and native
30 Hz clock as the marginal experiment. The arms are taken from the same frame
of the same episode before cropping; they are never independently shuffled or
temporally aligned after sampling. The episode/crop RNG is unchanged, so the
joint cache uses exactly the marginal caches' windows.

The original PyTorch architecture now has 48 input channels
`[q_left,q_right | v_left,v_right | a_left,a_right | jerk_left,jerk_right]`, a
16-dimensional Gaussian latent, and 183 original fingerprint targets
(`14*12+15`). There are no added task or handoff labels. The auxiliary target
includes the original per-joint descriptors and global descriptors computed
over all 12 joints. Training retains 60 epochs, 800 steps per epoch, batch 128,
six crops plus the full window, the original optimizer/loss settings, and
120,000 training-only reference segments split equally between sources. The
same validation rule chooses the checkpoint. No joint posture model is fitted.

`work/yam/models/bimanual/` contains `encoder.pt`, `encoder.npz`, configuration,
reference/validation reports, checksums and resume state. These are local model
artifacts, excluded from Git. The checkpoint retains the original PyTorch class
and historical reference-statistic field names. `pipeline_bimanual.json` and
its own lock track this run without replacing the marginal pipeline status.

## Controlled three-condition demo

```sh
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 JAX_PLATFORMS=cpu \
  uv run --extra planning --extra simulation python examples/yam_bimanual.py --compare
```

The task is the direct-transfer, three-cube handoff scene. All three conditions
use one 12-joint path solve and one shared phase clock, with the same
initialization, Cartesian handoff tracking, motion limits, collision geometry,
clearance margins, controller and ownership barriers:

| Condition | Posture selection | Motion objective | Timing |
| --- | --- | --- | --- |
| `off` | Nearest feasible IK | Smoothness and acceleration only | Fixed |
| `independent` | Separate left/right GMMs | Mean of the two marginal encoder costs | Shared clock optimized with that mean |
| `bimanual` | The same separate GMMs | One encoder on the synchronized 12-joint motion | Shared clock optimized with the joint cost |

The independent adapter averages the two marginal Mahalanobis costs so its
nominal reference cost is 16, like the joint 16-dimensional encoder. It has no
learned cross-arm latent. It supports both the path objective and the masked
model-rate timing objective. This comparison uses a common planner/clock for
all variants; the independent condition therefore differs from the earlier
demo that optimized and retimed the two arms separately.

Every condition has the same budget of 320 optimizer iterations per attempt,
with one warm-start retry if a finite candidate fails the constraints. Rejected
candidates are never executed; a retry must pass the original tolerances.

An arm without a motion in the task phase is constrained to hold its pose. The
other arm's moving geometry and the carried cube participate in the joint
collision constraints, including derivatives through both chains. A cube's
rigid hand-relative transform is used only for planning geometry; physics
replay continues to use free-body contact grasps. These remain sampled local
constraints, rather than a continuous collision certificate or a global search
planner. Every replay checks physical grasps, receiver support before giver
release, final placement, and unintended robot contacts.

Outputs under `work/yam-bimanual-demo/`:

- `off_vs_bimanual.mp4`: DATAFARM off on the left, joint encoder on the right.
- `all_three.mp4`: off, independent encoders, joint encoder, left to right.
- `comparison.json`: measured replay metrics for each available condition.
- `{off,independent,bimanual}/video.mp4`, `metrics.json`, `simulation.npz`:
  normal-speed individual replays and measurements.
- Each condition's `plan.npz`, `source_paths.npz` and `planning.json`: emitted
  commands, untimed 12-joint phase paths, shared timing records, IK records and
  model/scene hashes.

Comparisons default to **2× playback**, labeled in the top right. Each panel
retains its actual task timing; finished panels hold their last frame. Use
`--speedup 1` or `--speedup 4` to change playback, `--no-render` for a physics
preflight, and `--reuse-plan` to rerender validated plans. `--variant` runs one
condition. `--no-retime` makes all clocks fixed for a path-only ablation.

## Local validation run

The default training run selected epoch 57 of 60 and retained the marginal
corpus manifest `0140e9c39bfcb49e7c66f2ed469d29d0778b607f8e4c6f454496bfd3898a35e0`.
The checkpoint loaded strictly into the pinned upstream encoder class; maximum
latent disagreement with the JAX export was below 2.3e-6 on six held-out windows
from both sources. Original posture artifact checksums still matched.

The default-seed scene produced these measured replay results (durations include
the common 1.2-second final settling interval):

| Condition | Duration | Handoffs / placed | Unintended robot contacts beyond 1.5 mm |
| --- | ---: | ---: | ---: |
| Off | 44.90 s | 3 / 3 | 0 |
| Independent encoders | 69.25 s | 3 / 3 | 0 |
| Bimanual encoder | 60.83 s | 3 / 3 | 0 |

The joint run needed one warm-start retry on its first presentation phase;
every accepted source path satisfied the original constraint tolerance. Idle
arms were held exactly, tracked handoff legs deviated by at most 0.55 mm, and
maximum final placement error was 6.1 mm across the three conditions. Full local
records are in `work/yam/bimanual_validation.json` and
`work/yam-bimanual-demo/validation.json`.
