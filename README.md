# DataFarm-PyRoki

A JAX port of DATAFARM's frozen motion encoder with pretrained weight loading, plus a
whole-motion-leg residual and constrained trajectory solver for PyRoki, followed
by JAX stroke retiming and joint-density-based IK selection.

## Setup

Python and dependencies are managed with **uv**. The lockfile pins PyRoki and
jaxls to tested Git revisions.

```sh
uv sync --extra planning
# Reproduce the pinned checkpoints locally (downloads the upstream source assets):
uv run python scripts/convert_checkpoint.py
uv run python scripts/convert_joint_prior.py
uv run python examples/score_motion.py
uv run --extra planning pytest
```

After obtaining the converted weights, encoder-only inference needs neither PyTorch
nor the planning dependencies:

```sh
uv sync --no-dev
uv run --no-dev python examples/score_motion.py
```

Checkpoint binaries are **excluded from Git and package builds**. Reproduction
above writes ignored local NPZ files under `src/datafarm_pyroki/assets/`. Alternatively,
download converted artifacts from a separate host; the fetch helper accepts Hugging
Face `resolve` URLs and verifies SHA256 against the tracked provenance:

```sh
uv run --no-dev python scripts/fetch_assets.py \
  --encoder-url "$ENCODER_URL" --joint-prior-url "$JOINT_PRIOR_URL" \
  --destination "$HOME/.local/share/datafarm-pyroki"
export DATAFARM_ASSET_DIR="$HOME/.local/share/datafarm-pyroki"
```

Set the two URLs to the published artifact locations. No Hugging Face repository
is assumed. Explicit checkpoint paths take precedence over `DATAFARM_ASSET_DIR`;
otherwise loaders use the local source asset folder. Provenance and license notices
remain tracked. Inference needs no CUDA or upstream DATAFARM/cuRobo installation.
The default JAX installation uses CPU.

## Encoder API

```python
import jax
import jax.numpy as jnp
from datafarm_pyroki import load_encoder

encoder = load_encoder()  # or load_encoder("path/to/another_model.npz")
q = jnp.zeros((32, encoder.spec.n_joints))  # example only, radians, MODEL joint order
latent = encoder.latent(q, source_dt=0.15)
residual = encoder.residual(q, source_dt=0.15, weight=1.0)
cost, gradient = jax.jit(
    jax.value_and_grad(lambda positions: encoder.cost(positions, source_dt=0.15))
)(q)
```

Inputs are `[time, joints]` or `[batch, time, joints]`. `source_dt` describes the
uniform waypoint clock and is a static Python scalar under JIT. The encoder
resamples at its training rate, constructs `[q | velocity | acceleration | jerk]`,
normalizes the channels, and returns a deterministic latent mean. The squared
norm of `residual` equals `weight * Mahalanobis²`; there is one latent per leg.

The default reference model is **DROID Franka**, with seven arm joints ordered
proximal to distal, a 15 Hz rate, and a 16-dimensional latent. Its canonical coordinate names
are `joint1` through `joint7`; these describe training order, not URDF aliases.
Gripper coordinates are not encoder inputs. The general loader and network infer
joint and latent dimensions from each artifact. `EncoderSpec` records identity,
ordered joints, and rate; `JointMap` maps them explicitly into a robot model.
Matching DOF counts alone do not establish model compatibility.

## Add the learned cost to an existing PyRoki problem

```python
from datafarm_pyroki import JointMap, load_encoder
from datafarm_pyroki.integration import make_motion_cost

encoder = load_encoder()
# `robot`, `timesteps`, `costs` come from your existing PyRoki trajectory problem.
# If URDF names differ, provide aliases {model_joint_name: robot_joint_name}.
joint_map = JointMap.from_names(
    robot.joints.actuated_names,
    encoder.spec.joint_names,
    aliases={f"joint{i}": f"panda_joint{i}" for i in range(1, 8)},
)
waypoints = tuple(robot.joint_var_cls(t) for t in range(timesteps))
costs.append(
    make_motion_cost(
        encoder,
        waypoints,
        joint_map=joint_map,
        source_dt=0.15,
        weight=1.0,
    )
)
# Add your collision, endpoint, task and smoothness terms, then solve as usual.
```

Pass **individual scalar-ID variables** as one tuple: this creates one factor
spanning the leg. Passing a batched timestep variable alone would make jaxls treat
timesteps as separate examples. The factory rejects that mistake. It uses
reverse-mode Jacobians and does not optimize the encoder parameters.

`weight` multiplies the squared cost, not the residual. Start by checking the scale
of this term against your base objective; DATAFARM's cuRobo weight does not carry
over numerically to PyRoki's differently scaled costs.

## Refine a complete leg

`datafarm_pyroki.integration.solve_leg` accepts a PyRoki robot, actual start and
selected goal joint configurations, an explicit `JointMap`, and an optional initial
path. It includes joint and velocity constraints, endpoint constraints, smoothness
and acceleration costs, optional acceleration limits, optional self-collision
constraints and swept-capsule world-collision constraints. A callback adds your
own task and attached-object costs/constraints. It checks the resulting constraint
residuals and raises `InfeasibleTrajectoryError` if they exceed tolerance.

```python
from datafarm_pyroki.integration import solve_leg

result = solve_leg(
    robot,
    encoder,
    q_start,
    q_goal,
    joint_map=joint_map,
    source_dt=0.15,
    initial_waypoints=seed_path,
    robot_coll=robot_collision_model,
    world_coll=scene_obstacles,
    encoder_weight=1.0,
)
q_optimized = result.positions  # robot joint order; clock remains source_dt
```

The CLI in `examples/optimize_leg.py` loads a local URDF and saves a motion array.
Use `uv run --extra planning python examples/optimize_leg.py --help` for arguments.
The CLI checks self collision only; add scene obstacles through the Python API.
This is local path refinement, so initialization matters. Self-collision checks
are at waypoints and swept world collision is a geometric approximation. The
returned constraint tolerance is not a continuous-time physical safety guarantee.
No examples execute a robot.

## Choose an IK endpoint by NLL

```python
from datafarm_pyroki import load_joint_prior, select_ik_solution

prior = load_joint_prior()  # separately obtained DROID joint-configuration GMM
selected = select_ik_solution(
    prior,
    ik_candidates,  # [..., candidates, robot_joints]
    joint_map=joint_map,
    feasible=ik_success_and_collision_mask,  # [..., candidates], supplied by your IK checks
    lower_limits=robot_joint_lower_limits,
    upper_limits=robot_joint_upper_limits,
)
q_goal = selected.positions  # full robot configuration, ready for solve_leg
```

`rank_ik_solutions` is the JIT-compatible counterpart. It returns stable sorted
indices and original-order NLL scores, with invalid candidates set to infinity.
`select_ik_solution` raises `NoFeasibleIKSolutions` if any query has no valid branch.
The score is a full Gaussian-mixture NLL in input joint units; it has the same
branch ranking as DATAFARM's constant-offset score. The GMM is distinct from the
encoder's latent Gaussian. Load another prior or construct
`JointPositionPrior.from_parameters(...)` for another trained model; see
[docs/selection.md](docs/selection.md). The reference GMM retains its upstream
NVIDIA research/evaluation license, described below.

## Retime a stroke after path optimization

```python
from datafarm_pyroki import retime_stroke

stroke = retime_stroke(
    encoder,
    joined_waypoints,
    joint_map=joint_map,
    original_duration=source_duration,
    output_dt=0.02,
    velocity_limits=joint_velocity_limits,
    acceleration_limits=joint_acceleration_limits,
    lead_speed=0.0,
    trail_speed=0.07,
)
# stroke.positions, velocities, accelerations, times, dt are ready for inspection.
```

This optimizes duration and speed-profile shape on a fixed geometric curve,
scoring at the model's training rate with differentiable duration masks. Defaults
are 64 clock knots, 16 simultaneous starts and 500 Adam steps. Motion limits apply
to every robot coordinate, including coordinates outside the encoder mapping.
Returned `dt` is the actual uniform timestep, approximately `output_dt`.

For a plan with several legs and gripper actions, let the helper form the strokes:

```python
from datafarm_pyroki import MotionLeg, GripperEvent, retime_plan

steps = [
    MotionLeg(retract.positions, retract.source_dt, "retract"),
    MotionLeg(transit.positions, transit.source_dt, "transit"),
    GripperEvent(command="close", gripper="tool", dwell_seconds=0.1),
    MotionLeg(next_motion.positions, next_motion.source_dt, "next motion"),
]
plan = retime_plan(
    encoder,
    steps,
    joint_map=joint_map,
    output_dt=0.02,
    velocity_limits=joint_velocity_limits,
    acceleration_limits=joint_acceleration_limits,
    boundary_speed=0.07,
)
# plan.steps retains the gripper events; plan.events attaches their new timestamps.
# plan.positions/times include stationary dwells, and limits are rechecked at junctions.
```

The helper preserves event payloads, identity, order and requested dwell lengths.
Boundary speeds are configurable soft targets, not predicted from gripper inputs.
A failed motion-limit or validation check raises `RetimingError`. An optional
`trajectory_validator(positions, times)` checks application-specific feasibility.
Geometry smoothing is off by default; enabling it needs renewed collision/task
validation. See [docs/retiming.md](docs/retiming.md) for clock, boundary and output
conventions. `examples/retime_motion.py` retimes saved motion arrays without robot
execution. Retiming and IK ranking do not require the planning extra or PyTorch.

## Relation to DATAFARM

The closest stage structure is:

1. Select feasible IK endpoints with a reference configuration prior.
2. Optimize each planner leg with the motion latent cost added to the base objective.
3. Join legs between gripper events and retime each complete stroke, using the same
   encoder plus a configured boundary-speed term.

This repository implements the learned encoder, GMM-based endpoint ranking,
PyRoki leg refinement and standalone stroke retiming. Task sequencing and IK
candidate generation stay with the caller. See [docs/integration.md](docs/integration.md)
for the upstream hooks. `solve_leg` keeps its clock fixed; its endpoint-rest
option applies before stroke retiming.

## Checkpoint provenance and reproduction

The local encoder artifact is
`src/datafarm_pyroki/assets/droid_encoder.npz`, with original source/checksums tracked in
`src/datafarm_pyroki/assets/provenance.json`. The joint prior has its own tracked
`joint_prior_provenance.json`. Neither binary is committed or packaged. All
state_dict tensors, including normalization buffers, latent log-variance and
auxiliary-head parameters, are preserved. The optional
conversion tool reads the verified upstream PyTorch checkpoint and writes a
pickle-free NPZ:

```sh
uv run python scripts/convert_checkpoint.py
uv run python scripts/convert_joint_prior.py
```

**The reference artifact's training metadata differs from the current upstream
README/training script.** The checkpoint says 30,000 DROID training episodes plus
cuTAMP motion, feature regression and a Fisher-discriminant objective, selected
by held-out pair silhouette. Its reference latent statistics are DROID-derived.
We port that exact artifact without retraining; the metadata is retained verbatim.
See [docs/model.md](docs/model.md) for the architecture and conversion contract.

## Validation

```sh
uv run --extra planning pytest
uv run ruff check .
uv run ruff format --check .
```

Tests compare the released PyTorch model against JAX for latent means, log
variances, auxiliary outputs, padded masks, preprocessing, costs and waypoint
gradients. They exercise an actual constrained PyRoki solve, a single whole-leg
jaxls factor, joint reordering, and an alternate model dimension/rate. Timing
objective and profile/duration gradients are checked against PyTorch; emission,
stroke grouping, gripper holds, junction limits and GMM NLL selection are tested.
Synthetic alternate-model tests verify plumbing, not trained behavior on another embodiment.

The first solve compiles the network and its trajectory Jacobian. A whole-leg
factor couples all its waypoints, so its Jacobian is dense within that leg.

## License

Project code and the DATAFARM encoder/checkpoint are MIT-licensed; see
[LICENSE](LICENSE), [licenses/DATAFARM.txt](licenses/DATAFARM.txt) and
[licenses/TIPTOP.txt](licenses/TIPTOP.txt) for attribution.

The separately obtained **joint GMM asset** (`droid_joint_prior.npz`) is extracted from cuTAMP
and remains under its [NVIDIA License](licenses/CUTAMP.txt), which limits use to
non-commercial research or evaluation. That asset has separate provenance and
license notices alongside the local/downloaded weights. It does not relicense your
independently trained priors. The code package is MIT-licensed. PyRoki and jaxls
remain separate optional dependencies under their respective licenses.
