# Integration boundaries

## Upstream implementation

DATAFARM pins these implementations:

- [DATAFARM](https://github.com/SamratSahoo/DATAFARM/tree/fd403ea6eb16163261a13439fe9bbc10a980f50d)
- [TiPToP](https://github.com/SamratSahoo/tiptop/tree/013c27a48389db70208f9edb696d137dd5137f02)
- [cuTAMP](https://github.com/SamratSahoo/cuTAMP/tree/3a2e4d000339f7460f1989bde84d32b328ac92f9)
- [cuRobo](https://github.com/SamratSahoo/curobo/tree/3a90ff49eee169d9636b2a679d98457a2592fb52)

TiPToP's `motion_planning.apply_cost_overrides` maps `encoder_weight` and
`encoder_path` into `vae_manifold_cfg` of the gradient trajectory configuration,
then constructs `MotionGen` with that configuration. cuRobo's `ArmReacher` creates
`VaeManifoldCost`, evaluates it on rollout joint positions and adds the result to
its cost list. That class computes one Mahalanobis score per leg and spreads it
across the horizon so the sum gives the intended weighted score. Gradients flow
through the frozen network into the motion optimizer. It is not post-hoc ranking.

cuTAMP's particle initializer separately asks for multiple endpoint IK solutions,
then `select_posture_branch` ranks them with the configuration prior in
`posture_prior.py`. The primary prior is a GMM; an older pairwise model is retained
as a fallback. Those parameters live in cuTAMP's `posture_ref.npz`, separately
from the encoder checkpoint. This repository provides a separate GMM converter and
`select_ik_solution` for the same ranking. The resulting external artifact retains
its distinct upstream license.

## Equivalent PyRoki hook

Add `make_motion_cost` to the same `jaxls.LeastSquaresProblem` that optimizes a
leg's joint variables. The new residual depends on the entire leg and uses
reverse-mode autodiff through the JAX encoder. Preserve the geometric/task
constraints that defined that leg. The fixed clock matches the normal upstream
path-optimization mode. Do not optimize a full gripper-to-gripper trajectory segment as a
single geometry problem if reproducing DATAFARM's decomposition is the goal.

The included `solve_leg` is a reference refinement solver. It takes previously
selected joint endpoints, never re-solves IK silently, and accepts a seed path.
By default it pins the first two and last two positions to the endpoints, matching
a leg that begins and ends at rest. Set `rest_endpoints=False` only when your
surrounding pipeline supplies a different boundary convention.

For repeated solves, create a `make_leg_solver` once. Its callable accepts new
endpoints, initial waypoints and a runtime `context` PyTree. An
`extra_costs(waypoints, context)` callback adds task constraints such as changing
TCP targets. Calls with the same shapes/dtypes reuse the compiled optimizer.
`solve_leg` uses the same construction and retains its existing
`extra_costs(waypoints)` callback for one-shot use. Weights multiply squared costs;
`encoder_weight=0` supplies the corresponding baseline solver.
Returned boundary positions are exact. All constraints are rechecked after
restoring those boundaries, including adjacent velocity/collision constraints.
An `InfeasibleTrajectoryError` can carry `positions` and
`max_constraint_violation` for diagnostics or a warm-start retry. Those positions
are rejected candidates; a subsequent solve must validate them before execution.

```python
from datafarm_pyroki.integration import make_leg_solver

solve = make_leg_solver(
    robot,
    encoder,
    joint_map=joint_map,
    source_dt=0.1,
    timesteps=20,
    rest_endpoints=False,
)
first = solve(q_start, q_middle)
second = solve(q_middle, q_goal)
```

The convenience solver's smoothness/acceleration costs and augmented-Lagrangian
constraints are not cuRobo's full objective or optimizer. Weights, initialization,
collision representations and convergence differ. Equivalent motion-style costs
do not imply identical paths or identical downstream training results.

## IK candidate generation

`make_ik_solver` provides batched PyRoki IK independently of endpoint selection.
Supply the target link, an SE3 target `[w,x,y,z,px,py,pz]` and a seed bank in
actuated robot joint order. Each candidate is checked for finite coordinates,
joint limits and pose error. A caller-supplied `feasibility(q)` callback adds
scene collision/task checks. Candidate order is preserved, including infeasible
results, so ranking and failure handling remain explicit.

```python
from datafarm_pyroki import select_ik_solution
from datafarm_pyroki.integration import make_ik_solver

ik = make_ik_solver(robot, "tcp")
candidates = ik(target_pose, ik_seeds, feasibility=collision_free)
selected = select_ik_solution(
    prior,
    candidates.positions,
    joint_map=joint_map,
    feasible=candidates.feasible,
)
```

The factory reuses compiled code across targets with matching shapes. It does not
sample grasps or seeds, assume a Franka joint count, or choose an IK branch.

## Sampling and validating a complete plan

`interpolate_plan(steps, output_dt=...)` joins motion legs between gripper events
and samples C2 joint splines. `sample_plan` additionally applies fixed-duration
boundary easing and stationary gripper dwells, returning a `PlanTimeline` with
`positions`, `times` and timed `events`. The initial configuration comes from
the first motion leg; an event-only plan requires an explicit `initial_position`.

`interpolate_trajectory` and `validate_trajectory` also operate on standalone joint
arrays and explicit timestamps. Validation accepts joint-ordered position,
velocity and acceleration limits and checks interval speeds as well as sampled
derivatives. Neither utility depends on PyRoki or MuJoCo. Interpolation can change
geometry between source waypoints, so scene/task checks remain caller-provided.

## Joint mapping and model selection

Select a model for the training embodiment, then construct a `JointMap` from the
PyRoki robot's `robot.joints.actuated_names` to `encoder.spec.joint_names`. Aliases
are explicit. The mapping selects only encoder coordinates; all robot variables
still participate in your task/collision constraints. It can also represent a
model trained on an ordered subset of coordinates, when that is intentional.

`solve_leg` checks that the map belongs to its supplied robot and encoder. It
never falls back to the first seven coordinates. Load another artifact to change
the model; its input width, latent dimension, rate and reference distribution
are independent. The artifact must have been trained for the coordinates and
embodiment you choose; names/metadata are an interface contract, not retargeting.

## Trajectory segment retiming

TiPToP calls `blend_cutamp_plan` after `run_cutamp` returns, before export or
execution. Group consecutive motion legs between gripper actions. Preserve the
gripper steps in sequence. Each group becomes one trajectory segment whose geometric path is
held fixed after optional smoothing; start/end positions remain pinned.

The upstream `encoder_retime_group` optimizer changes the trajectory segment duration and
time allocated across 64 arc-length knots. It uses 16 batched starts and Adam,
with a shared frozen encoder/reference Gaussian. Score motions on a real model-rate
grid with a differentiable soft validity mask, not by holding sample count fixed
and pretending the encoder's filter durations change. The padded encoder path in
this package supplies the matching prefix-masked inference operation.

The objective combines latent Mahalanobis cost with a squared speed-target term
at gripper-adjacent boundaries. Those targets come from configuration; they are
not outputs of a gripper-conditioned learned model. The default reference target
is the latent cluster mean; the code also has an optional sampled-target mode.
Candidates are emitted at the controller step and rejected if velocity or
acceleration caps fail. Event order and gripper actuation remain separate.

The standalone JAX implementation is `retime_trajectory_segment`; `retime_plan` groups legs,
retains events/dwells, retimes trajectory segments and checks the full stitched signal. It
loads no TiPToP or cuRobo modules. It uses each encoder's sampling rate and joint
mapping, with limits on all robot coordinates. The encoder is fixed during Adam;
only profile knots and duration change.

Compared with the pinned implementation, smoothing defaults to zero and the
original continuous polyline is retained separately from the 64 clock knots.
Initial clock candidates are kept as feasible fallbacks, and final candidates
are ranked by their emitted style, boundary and regularization objective. The
default clock is monotone C2, with true easing into requested rest endpoints;
profile variation, duration, acceleration and jerk penalties discourage uneven
timing. These are explicit engineering priors added to the upstream style objective.
Rest-end acceleration
is recomputed after endpoint velocity pinning; the whole plan is checked across
gripper junctions. Sampled targets use an explicit JAX PRNG key, rather than a
hidden geometry-derived seed. These are explicit adaptations, not a bitwise port
of the PyTorch optimizer. See [retiming.md](retiming.md).
