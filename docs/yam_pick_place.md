# Bimanual YAM pick-and-place comparison

```sh
uv sync --extra planning --extra simulation
MUJOCO_GL=egl uv run --extra planning --extra simulation python examples/yam_pick_place.py --compare
```

This example uses the trained models under `work/yam/models/{left,right}/` by
default. Use `--models` for another directory with the same arm-specific artifact
names; see [training](training.md). Each arm uses its own six-joint posture GMM,
24-channel encoder, normalization, and latent reference statistics. Model names,
ordered joints, and the native 30 Hz model clock are checked explicitly.

Two YAMs simultaneously move two cubes each onto a common tray. Both modes use
the same scene, seed bank, object positions, constraints, controller gains,
contact physics, and gripper dwell times. The baseline selects nearest feasible
IK branches and optimizes smoothness with a fixed trajectory clock. DATAFARM
selects feasible branches using the corresponding arm's GMM, adds that arm's
whole-leg encoder cost, and retimes its gripper-delimited segments. The default
learned-cost weight is 0.1; the constrained local solver has a 180-iteration
budget in both modes. `--no-retime` isolates posture/path effects.

Each arm plans independently in its side of the workspace, with joint/velocity,
table-clearance, and midline constraints. Contact legs retain Cartesian tracking;
transit legs fix their endpoints and leave intermediate poses free. MuJoCo
collision checks sample optimized paths densely. Synchronized playback then
checks actual inter-arm, robot/table, and robot/object contacts. This is a local
planner for this specified scene, not general obstacle search or task planning.

The single-arm timing API is applied separately to each marginal model. Timelines
are combined at a common ~20 ms output clock, preserving both grippers' event
times; an arm that finishes early holds its final configuration. Physics advances
at 2 ms. The video is 30 fps at normal speed, with the shorter comparison holding
its final frame. These clocks are distinct from the encoders' 30 Hz sampling rate.

Robot geometry comes from the pinned MIT-licensed I2RT YAM linear-4310 model in
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/cda5236f98383f6fbaa4820c6b18dba892cd087f/i2rt_yam).
Meshes and license notices are downloaded into `work/yam-menagerie/` and checked
against Git blob hashes. The URDF for each arm is exported from the compiled
MuJoCo kinematics, and TCP agreement is verified before planning. Both arms keep
their native I2RT joint conventions; the right arm is not mirrored in joint space.

Grasps use the physical parallel-jaw pads and friction. All four cubes have free
joints: there are no object welds, attachments, or pose overrides during playback.
The demo requires bilateral finger contact and a lift above 9 cm for every cube,
four final placements within 3.5 cm of their targets, and no unintended robot
penetrations deeper than 1.5 mm. Those measured checks are recorded rather than
inferred from the planned end-effector path.

Outputs under `work/yam-pick-place/`:

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
