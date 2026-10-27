# MuJoCo pick-and-place comparison

`examples/pick_place.py` uses PyRoki IK and trajectory optimization to move three
4 cm cubes from a table onto a flat disc. The simulated robot is a nominal Franka
Panda arm with a Robotiq 2F-85 gripper, matching the DROID hardware layout. Object
poses and the task sequence are specified by the example; it does not use perception.

The example file shows the pick/place sequence and the steps to run it. Supporting
modules contain the [planner](../examples/_pick_place_planning.py),
[scene](../examples/_pick_place_scene.py), [timing](../examples/_pick_place_timing.py),
[simulation and video](../examples/_pick_place_simulation.py), and
[CLI and saved-plan handling](../examples/_pick_place_demo.py).
The planner uses the package's reusable `make_ik_solver` and `make_leg_solver`;
only this scene's collision policy and contact/table costs stay in the example.
Interpolation, fixed-clock sampling and limit checking use `interpolate_plan`,
`sample_plan` and `validate_trajectory`. See [integration](integration.md) for these
APIs. The shared [playback helper](../examples/_simulation.py) accepts explicit
joint/actuator/body mappings and callbacks for gripper events, observations and
frame annotations. Cube contact rules and success checks remain in this demo.
Video encoding uses ImageIO's PyAV backend; the simulation extra installs it
without requiring a separate FFmpeg executable.

```sh
uv sync --extra planning --extra simulation
uv run --extra planning --extra simulation python examples/pick_place.py --compare
```

Run a single mode with `--datafarm` or `--no-datafarm`. `--compare` runs both and
writes a synchronized, normal-speed split-screen video. The shorter run holds its
last frame after finishing; neither run is sped up to match the other.
Learned retiming is enabled by default for DATAFARM (`--retime`). Use `--no-retime`
to give both modes the same fixed trajectory segment clock and isolate posture selection and
geometric trajectory optimization. The baseline always uses the fixed clock.

The default output folder is `work/pick-place-comparison/`:

- `baseline/video.mp4`, `datafarm/video.mp4`: annotated individual videos.
- `comparison.mp4`: baseline on the left, DataFarm on the right.
- `comparison.json`: measured execution and motion metrics for both modes.
- Each mode's `plan.npz`, `planning.json`, `simulation.npz`, and `metrics.json`:
  commanded trajectory, selected IK branches, optimization/retiming diagnostics,
  actual joint and cube histories, and final placement results.
- `source_plan.npz`: each mode's smooth source trajectory before final timing
  (baseline boundary ramps or DATAFARM retiming);
  its gripper timestamps are recorded as `source_events` in `planning.json`.
- `scene.xml` and `robot.urdf`: the generated simulation and planning models.

Full-resolution videos, binary trajectories and downloaded assets remain in
ignored local `work/` directories. A compressed 480p comparison and its thumbnail
are included in `docs/assets/` for the README. The robot meshes are fetched once
from a pinned
[MuJoCo Menagerie revision](https://github.com/google-deepmind/mujoco_menagerie/tree/cda5236f98383f6fbaa4820c6b18dba892cd087f).
Downloads verify Git blob hashes. The upstream Franka Apache-2.0 and Robotiq BSD
license files are kept with the downloaded models. Mesh binaries are not committed.
The learned models use the package's pinned Hugging Face downloads and cache.

## What the flag changes

| Stage | DataFarm off | DataFarm on |
| --- | --- | --- |
| IK | Nearest feasible solution to the current configuration | Minimum posture-GMM NLL among feasible solutions |
| Trajectory optimization | Fixed joint endpoints, smoothness, acceleration, joint/velocity limits and table clearance; TCP tracking only for contact legs | The same terms plus one frozen encoder residual spanning each whole leg |
| Timing | Fixed source durations, continuous motion through leg joins; rest at gripper events and plan ends | Learned trajectory segment retiming by default; fixed clock with `--no-retime` |
| Gripper/physics | Identical opening/closing commands, dwell lengths and contact model | Identical opening/closing commands, dwell lengths and contact model |

The baseline loads the released models only to report comparable NLL/style scores.

Both modes use a fixed random seed for their common IK seed bank, plus the current
configuration as a warm start. The gripper TCP is exported from MuJoCo into the
URDF, and PyRoki/MuJoCo FK is checked before planning. Each of the 19 motion legs is
optimized in PyRoki/jaxls. IK candidates are checked for pose error, joint limits
and MuJoCo collision geometry. Optimized paths receive denser collision checks.
Descents to grasp/place and vertical lifts/retreats retain Cartesian tracking.
Approaches between objects, raised transfers and the return home fix only their
joint endpoints. Their intermediate positions and orientations are optimization
variables, with joint/velocity limits, arm/table clearance, and a 10 cm lower
bound on TCP height. No straight-line pose target or reference-path cost is added
on those free-space legs. Dense MuJoCo collision checks and the physical replay
validate the resulting paths. This is local trajectory optimization from a joint
interpolation seed, not a general obstacle-search/task-planning system.
Each leg's metadata records its `motion_type`, fixed joint waypoint indices and
deviation from the straight Cartesian reference. On free-space legs that
reference is diagnostic only; `max_tracking_error_m` is null.

Intermediate leg endpoints are position constraints, with no duplicated rest
samples or per-leg minimum-jerk clock. Both modes interpolate a common type of
C2 cubic joint curve through the source waypoints, with zero endpoint velocity
only at gripper events and the plan ends. The baseline preserves the original
trajectory segment duration (the sum of 1.9 s source legs) and samples this curve at 20 ms.
Its fixed clock uses smooth speed ramps over the first/last 10% of each trajectory segment,
with uniform cruise between them. These ramps only occur at gripper/plan
boundaries; intermediate legs do not receive individual easing or holds.
Geometry between source
waypoints changes slightly; the physics replay checks unintended contacts.

With `--retime`, DataFarm mode retimes seven gripper-delimited trajectory segments with 16 clock
knots, eight starts and 250 Adam iterations. `--retime-starts` and
`--retime-iterations` change these settings. Timing uses a fixed geometric path,
zero boundary-speed targets for grasps/releases, and checks commanded velocities,
accelerations and stitched junctions. The demo's retiming caps are 2 rad/s and
20 rad/s² per arm coordinate, with duration bounds from 0.7× to 3× the original
trajectory segment, clipped by the retimer's overall duration range. These are simulation
settings. `--encoder-weight` changes the whole-leg learned-cost weight (default 0.3).
The clock is a monotone C2 quintic with genuine easing at rest boundaries. Explicit
penalties discourage adjacent clock-rate fluctuations (weight 0.1), large departures
from uniform arc speed (0.5), and excess duration (2 times duration/source duration).
Direct mean squared acceleration (0.1) and jerk (0.01) penalties also account for
bends in the joint path; smooth clock rates alone cannot remove that roughness.
These additions trade some learned style score for smoother, more efficient timing;
they are configurable through `RetimeConfig` and recorded with each segment.
See [retiming](retiming.md) for the complete objective and geometry limitations.

## Actual manipulation and validation

Arm position actuators execute the planned commands at MuJoCo's 2 ms physics
step, with gravity compensation. The Robotiq linkage closes using its actuator;
friction/contact forces lift each cube. Cube poses are initialized once and then
integrated by MuJoCo. There are no cube welds, attachments or pose overrides during
execution. The example records peak lifted heights, bilateral finger-contact
steps and unintended robot contacts. It fails if fewer than three cubes finish
on the disc or if unintended penetrating robot contacts are detected.

A fresh run measures both planner-space and executed behavior: per-leg latent
Mahalanobis costs, selected posture NLL, seven trajectory segment timing scores, execution
duration, joint tracking error, joint motion and final XY placement errors.
Commanded acceleration and jerk RMS are also recorded to inspect timing smoothness.
The style score is the objective being optimized; its reduction alone does not
establish greater human preference or task performance. A paired scene/seed shows
the full integration's behavior, not a separate ablation of its three components.

## Iteration and rendering

```sh
# Contact simulation and metrics without an OpenGL context:
uv run --extra planning --extra simulation python examples/pick_place.py --compare --no-render

# Render/simulate already saved plans without repeating IK/trajopt/retiming:
uv run --extra planning --extra simulation python examples/pick_place.py --compare --reuse-plan
```

When reusing plans, keep `--retime`/`--no-retime` consistent with the saved run.
Saved plan metadata uses `trajectory_segments`; regenerate older plans before
using `--reuse-plan`.

`--output`, `--asset-cache`, `--fps`, `--width`, and `--height` configure local
artifacts and rendering. On a headless Linux machine, use MuJoCo's supported EGL
or OSMesa backend via `MUJOCO_GL`. On macOS, offscreen rendering needs an available
graphics session. The no-render mode only requires the physics engine.
