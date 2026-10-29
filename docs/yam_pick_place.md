# Bimanual YAM handoff comparison

```sh
uv sync --extra planning --extra simulation
MUJOCO_GL=egl uv run --extra planning --extra simulation python examples/yam_pick_place.py --handoff --compare
```

This example uses the trained models under `work/yam/models/{left,right}/` by
default. Use `--models` for another directory with the same arm-specific artifact
names; see [training](training.md). Each arm uses its own six-joint posture GMM,
24-channel encoder, normalization, and latent reference statistics. Model names,
ordered joints, and the native 30 Hz model clock are checked explicitly.

Three cubes start on the left arm's side; the tray is entirely on the right
arm's side. The left arm picks and presents each cube at the center, and the right
arm acquires it and places it on the tray. The hands approach from opposite
directions and grip orthogonal cube faces, leaving space for both physical
grippers. The robot retains its native joint conventions throughout.

The sequence waits for the receiver to reach the cube and close for 0.65 seconds
before opening the giver. At execution time, giver release is blocked unless
the receiver has bilateral contact. The giver withdraws before the receiver
transports the cube; giver return and receiver placement can run concurrently.

Both modes use
the same scene, seed bank, object positions, constraints, controller gains,
contact physics, and gripper dwell times. The baseline selects nearest feasible
IK branches and optimizes smoothness with a fixed trajectory clock. DATAFARM
selects feasible branches using the corresponding arm's GMM, adds that arm's
whole-leg encoder cost, and retimes its synchronized task phases. The default
learned-cost weight is 0.1; the constrained local solver has a 180-iteration
budget in both modes. `--no-retime` isolates posture/path effects.

Each arm plans independently in its side of the workspace, with joint/velocity,
table-clearance, and midline constraints. Contact legs retain Cartesian tracking;
transit legs fix their endpoints and leave intermediate poses free. MuJoCo
collision checks sample optimized paths densely. Synchronized playback then
checks actual inter-arm, robot/table, and robot/object contacts. This is a local
planner for this specified scene, not general obstacle search or task planning.

To experiment with direct motions, without pregrasp, postgrasp lift, preplace,
or postplace retreat waypoints:

```sh
MUJOCO_GL=egl uv run --extra planning --extra simulation python examples/yam_pick_place.py --handoff --direct-transfers --compare
```

This variant writes to `work/yam-handoff-direct/`. The grasp, handoff, and placement
endpoints remain fixed, while the approach/transport path and orientation are free.
It replaces the fixed transit-height constraint with signed-distance constraints
using the scene's actual MuJoCo collision primitives, including fingers, wrist
cameras, table, tray, other cubes, and the other arm's current configuration.
Self collision has a 1 mm margin; block clearances use 20 mm to accommodate tracking
and placement errors, and other robot clearances use 3 mm. A rigid planning
transform carries the grasped cube, with separate payload/obstacle constraints;
payload/table and payload/tray contact is permitted at zero clearance. Only closed
fingers may contact the active cube. The optimizer checks waypoints and three
interior samples per interval, followed by denser path checks. Both modes receive
the same constraints and a 320-iteration budget. No fixed via pose is inserted to
lift over an obstacle. The short acquisition/withdrawal legs and ownership barriers
at the handoff remain. Simulation still uses only physical free-body grasps.

These are local, sampled geometry constraints, rather than a continuous collision
certificate or a global obstacle-search planner. Concurrent arms are checked
together during physical replay; each arm's solve treats the other as stationary.

The single-arm timing API is applied separately to each marginal model within
each shared phase. The phase lasts until both arms finish, then its gripper
events execute; an arm that finishes early holds position. Retiming therefore
preserves receiver-close, giver-release, and withdrawal barriers. A uniform
phase dilation, shared by both modes, enforces the 2 rad/s and 20 rad/s² caps if
interpolation would exceed them; its factor is recorded in `planning.json`.
Commands use a common ~20 ms output clock. Physics advances
at 2 ms. The video is 30 fps at normal speed, with the shorter comparison holding
its final frame. These clocks are distinct from the encoders' 30 Hz sampling rate.

Robot geometry comes from the pinned MIT-licensed I2RT YAM linear-4310 model in
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/cda5236f98383f6fbaa4820c6b18dba892cd087f/i2rt_yam).
Meshes and license notices are downloaded into `work/yam-menagerie/` and checked
against Git blob hashes. The URDF for each arm is exported from the compiled
MuJoCo kinematics, and TCP agreement is verified before planning. Both arms keep
their native I2RT joint conventions; the right arm is not mirrored in joint space.

Grasps use the physical parallel-jaw pads and friction. All three cubes have free
joints: there are no object welds, attachments, or pose overrides during playback.
The demo requires bilateral finger contact and a lift above 9 cm for every cube,
three final placements within 3.5 cm of their targets, and no unintended robot
penetrations deeper than 1.5 mm. Those measured checks are recorded rather than
inferred from the planned end-effector path. Handoff checks additionally record
simultaneous bilateral contact in both hands, bilateral receiver contact at
giver release, and receiver-supported airborne steps after the giver opens.

Outputs under `work/yam-handoff/`:

- `comparison.mp4`: baseline left, DATAFARM right, at the same wall-clock time.
- `comparison.json`: actual manipulation, timing, tracking, and learned-cost metrics.
- `{baseline,datafarm}/video.mp4`, `metrics.json`, `simulation.npz`: individual
  videos, physical success checks, actual/commanded joints, and object histories.
- Each mode's `plan.npz` and `planning.json`: the combined commands, gripper
  events, per-arm posture/path/retiming records, and model/scene hashes.
- `left.urdf`, `right.urdf`, `scene.xml`: generated kinematic/physics models.

Use `--no-render` to validate physics first. `--reuse-plan` simulates/renders saved
plans and rejects changed scene, model, or planning settings. `--fps`, `--width`,
`--height`, `--retime-starts`, and `--retime-iterations` configure rendering and
timing optimization. The displayed style score is an optimized model objective;
its reduction alone does not demonstrate human preference or broader task success.

The earlier independent four-cube comparison remains available by omitting
`--handoff`; it writes to `work/yam-pick-place/` and lets the arms finish on
independent clocks. The handoff and independent modes retain separate artifacts.
