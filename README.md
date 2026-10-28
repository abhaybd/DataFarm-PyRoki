# DataFarm-PyRoki

[Original DATAFARM repository](https://github.com/SamratSahoo/DATAFARM) ·
[Paper](https://arxiv.org/abs/2609.12316) ·
[Project website](https://prpl-group.com/datafarm/)

A JAX port of DATAFARM's frozen motion encoder with pretrained weight loading, plus a
whole-motion-leg residual and constrained trajectory solver for PyRoki, followed
by JAX trajectory segment retiming and joint-density-based IK selection.

https://github.com/user-attachments/assets/6ad5163d-79ff-4676-b148-c5c29aa86c72

## Getting Started

Plan between two end-effector poses on a Franka using an existing PyRoki `robot`.
Supply the end-effector link name `ee_link`, two poses `start_pose` and `goal_pose`
as `[w, x, y, z, px, py, pz]` (unit quaternion, meters), an `ik_seeds` array shaped
`[candidates, robot_joints]`, and per-joint `acceleration_limits` in rad/s².
Joint arrays use actuated robot order; seed positions are in radians.

```sh
uv sync --extra planning
```

```python
from datafarm_pyroki import JointMap, load_encoder, load_joint_prior
from datafarm_pyroki import retime_trajectory_segment, select_ik_solution
from datafarm_pyroki.integration import make_ik_solver, solve_leg

encoder, prior = load_encoder(), load_joint_prior()
joint_map = JointMap.from_names(robot.joints.actuated_names, encoder.spec.joint_names)
ik = make_ik_solver(robot, ee_link)


# 1. Solve IK and rank feasible solutions with the learned posture prior.
def ranked_ik(pose):
    candidates = ik(pose, ik_seeds)
    return select_ik_solution(
        prior, candidates.positions, joint_map=joint_map, feasible=candidates.feasible
    ).positions


# 2. Plan a joint-space path between the selected endpoints.
leg = solve_leg(
    robot,
    encoder,
    ranked_ik(start_pose),
    ranked_ik(goal_pose),
    joint_map=joint_map,
    source_dt=0.1,
)

# 3. Retime the path with the learned motion cost and joint motion limits.
trajectory = retime_trajectory_segment(
    encoder,
    leg.positions,
    joint_map=joint_map,
    original_duration=(len(leg.positions) - 1) * leg.source_dt,
    output_dt=0.02,
    velocity_limits=robot.joints.velocity_limits,
    acceleration_limits=acceleration_limits,
)
```

`trajectory.positions`, `.velocities`, `.accelerations` and `.times` contain the
retimed motion, starting and ending at rest. If URDF joint names differ from the
encoder's, pass `aliases=` to `JointMap.from_names`. For scene collision checks,
pass `feasibility=collision_free` to `ik` and `robot_coll`/`world_coll` to
`solve_leg`. See [planning utilities](docs/integration.md) for setup and options,
or the [pick-and-place demo](docs/pick_place.md) for a complete simulation.

## Setup

Python and dependencies are managed with **uv**. The lockfile pins PyRoki and
jaxls to tested Git revisions.

```sh
uv sync --extra planning
uv run python examples/score_motion.py
uv run --extra planning pytest
```

`load_encoder()` and `load_joint_prior()` download their default weights from:

- [abhaybd/DataFarm-Jax-MotionEncoder-Franka](https://huggingface.co/abhaybd/DataFarm-Jax-MotionEncoder-Franka)
- [abhaybd/DataFarm-Jax-PosturePrior-Franka](https://huggingface.co/abhaybd/DataFarm-Jax-PosturePrior-Franka)

Downloads use pinned commit revisions, verify SHA256 against the released
artifacts, and are reused from the standard Hugging Face cache. Both models are
public and download without authentication. Inference needs neither PyTorch nor
planning extras:

```sh
uv sync --no-dev
uv run --no-dev python examples/score_motion.py
```

To prefetch both models, choose a cache location, or load entirely offline:

```sh
uv run --no-dev python scripts/fetch_assets.py
# Optional cache override; the same location must be used when loading:
HF_HUB_CACHE="$HOME/.cache/datafarm-hub" uv run --no-dev python scripts/fetch_assets.py
```

```python
from datafarm_pyroki import load_encoder, load_joint_prior

encoder = load_encoder(local_files_only=True)  # requires an existing HF cache entry
prior = load_joint_prior(cache_dir="/path/to/hub-cache")
```

`HF_HUB_OFFLINE=1` also enables cache-only loading. Explicit NPZ paths still load
custom models directly: `load_encoder("my_encoder.npz")` or
`load_joint_prior("my_prior.npz")`. Default loading no longer checks
`DATAFARM_ASSET_DIR` or source-tree checkpoint files. Existing local artifacts stay
on disk and can be used through explicit paths.

Checkpoint binaries are **excluded from Git and package builds**. Provenance and
license notices remain tracked. Inference needs no CUDA or upstream DATAFARM/cuRobo
installation; the default JAX installation uses CPU.

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

For repeated legs, `make_leg_solver` creates a reusable compiled optimizer with
runtime task context. `make_ik_solver` generates candidates from caller-supplied
seeds and feasibility checks. Both live in `datafarm_pyroki.integration`; see
[reusable planning utilities](docs/integration.md).

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

prior = load_joint_prior()  # pinned DROID GMM from Hugging Face (cached after download)
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

## Retime a trajectory segment after path optimization

For fixed-duration motion, `sample_plan(steps)` joins legs, applies boundary easing
and preserves gripper events/dwells. `interpolate_plan` and `validate_trajectory`
provide interpolation and explicit motion-limit checks without a simulator.

```python
from datafarm_pyroki import retime_trajectory_segment

segment = retime_trajectory_segment(
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
# segment.positions, velocities, accelerations, times, dt are ready for inspection.
```

This optimizes duration and speed-profile shape on a fixed geometric curve,
scoring at the model's training rate with differentiable duration masks. Defaults
are 64 clock knots, 16 simultaneous starts and 500 Adam steps. Motion limits apply
to every robot coordinate, including coordinates outside the encoder mapping.
The default monotone C2 clock and explicit profile, duration, acceleration and
jerk penalties regularize the learned style objective. See [retiming settings](docs/retiming.md)
for weights, units and geometry requirements.
Returned `dt` is the actual uniform timestep, approximately `output_dt`.

For a plan with several legs and gripper actions, let the helper form the trajectory segments:

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
3. Join legs between gripper events and retime each complete trajectory segment, using the same
   encoder plus a configured boundary-speed term.

This repository implements the learned encoder, GMM-based endpoint ranking,
PyRoki leg refinement and standalone trajectory segment retiming. Task sequencing and IK
candidate generation stay with the caller. See [docs/integration.md](docs/integration.md)
for the upstream hooks. `solve_leg` keeps its clock fixed; its endpoint-rest
option applies before trajectory segment retiming.

## MuJoCo three-cube comparison

For the trained left/right YAM marginals, the
[bimanual YAM demo](docs/yam_pick_place.md) runs four physical cube transfers with
independent per-arm priors and a synchronized DataFarm off/on comparison video:

```sh
MUJOCO_GL=egl uv run --extra planning --extra simulation python examples/yam_pick_place.py --compare
```

```sh
uv run --extra planning --extra simulation python examples/pick_place.py --compare
```

This simulates a DROID-style Franka/Robotiq picking three cubes onto a flat disc
and renders normal-speed videos with DataFarm off/on. Use `--datafarm` or
`--no-datafarm` for individual runs. The flag switches posture-prior IK selection
and whole-leg encoder costs. DATAFARM also uses learned trajectory segment retiming by default;
use `--no-retime` to compare both modes on the fixed trajectory segment clock. Manipulation
uses MuJoCo contact physics; output includes grasp/placement checks, trajectories and comparison
metrics. See [docs/pick_place.md](docs/pick_place.md) for setup, controls and scope.

## Checkpoint provenance and reproduction

To train separate left/right YAM posture priors and PyTorch motion encoders from
a balanced ABC/MolmoAct2 mixture at their native 30 Hz, see
[YAM training](docs/training.md). This pipeline vendors the small upstream encoder
core, requires no DATAFARM fork or planner installation, and exports both original
and JAX artifact formats.

The default Hub revisions and SHA256 hashes are pinned in
`src/datafarm_pyroki/assets.py`. Original source/checksums are tracked in
`src/datafarm_pyroki/assets/provenance.json`; the joint prior has its own
`joint_prior_provenance.json`. Neither binary is committed or packaged. All
state_dict tensors, including normalization buffers, latent log-variance and
auxiliary-head parameters, are preserved. Optional conversion tools reproduce
pickle-free NPZs from verified upstream sources for development or custom use:


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

Tests and CI resolve their checkpoints from the same pinned public Hub
repositories without authentication or repository secrets. Hub resolution,
offline errors, explicit path overrides and checksum rejection have dedicated tests.

Tests compare the released PyTorch model against JAX for latent means, log
variances, auxiliary outputs, padded masks, preprocessing, costs and waypoint
gradients. They exercise an actual constrained PyRoki solve, a single whole-leg
jaxls factor, joint reordering, and an alternate model dimension/rate. Timing
objective and profile/duration gradients are checked against PyTorch; emission,
trajectory segment grouping, gripper holds, junction limits and GMM NLL selection are tested.
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

## Citation

If you use this port, please cite the software and the original
[DATAFARM paper](https://arxiv.org/abs/2609.12316).

```bibtex
@software{deshpande2026datafarmpyroki,
  author  = {Deshpande, Abhay},
  title   = {{DataFarm-PyRoki}: A {JAX} Port of {DATAFARM} for {PyRoki}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/abhaybd/DataFarm-PyRoki}
}

@article{sahoo2026datafarm,
  author        = {Sahoo, Samrat and Huang, Yixuan and Silver, Tom},
  title         = {{DATAFARM}: Distribution-Aligned Task and Motion Planning
                   for Fine-Tuning Vision-Language-Action Models},
  journal       = {arXiv preprint arXiv:2609.12316},
  year          = {2026},
  eprint        = {2609.12316},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2609.12316},
  url           = {https://arxiv.org/abs/2609.12316}
}
```
