"""Pick and place three cubes in MuJoCo with DATAFARM-guided motion planning.

Run: uv run --extra planning --extra simulation python examples/pick_place.py --datafarm
Add --compare to also run the baseline and save a side-by-side video.
"""

import json

import numpy as np
from _pick_place_demo import parse_args, run, validate_results
from _pick_place_planning import SOURCE_DT
from _pick_place_scene import (
    HOME,
    STARTS,
    TARGETS,
    build_scene,
    fetch_assets,
    initial_state,
    make_urdf,
)
from _simulation import compare_videos

from datafarm_pyroki import GripperEvent, MotionLeg


def plan_task(planner):
    """Build the pick/place sequence; Planner handles IK and learned motion costs."""
    steps = []
    current = HOME.copy()

    def move_to(position, label, *, contact=False):
        nonlocal current
        label = f"cube {cube + 1}: {label}"
        goal = planner.endpoint(position, current, allowed_cube=cube)
        path = planner.leg(current, goal, label, allowed_cube=cube, track_pose=contact)
        steps.append(MotionLeg(path, SOURCE_DT, label))
        current = goal

    for cube, (start, target) in enumerate(zip(STARTS, TARGETS, strict=True)):
        print(f"Planning cube {cube + 1}/3", flush=True)
        above_start, above_target = np.r_[start[:2], 0.22], np.r_[target[:2], 0.22]

        move_to(above_start, "approach")
        move_to(start, "descend to grasp", contact=True)
        steps.append(
            GripperEvent({"action": "close", "cube": cube}, gripper="robotiq", dwell_seconds=0.7)
        )
        move_to(above_start, "lift", contact=True)
        move_to(above_target, "transport")
        move_to(target, "lower onto disc", contact=True)
        steps.append(
            GripperEvent({"action": "open", "cube": cube}, gripper="robotiq", dwell_seconds=0.55)
        )
        move_to(above_target, "retreat", contact=True)

        # Future planning checks use this cube's new position.
        address = planner.model.joint(f"cube{cube}_free").qposadr[0]
        planner.scratch.qpos[address : address + 3] = target

    steps.append(MotionLeg(planner.leg(current, HOME, "return home"), SOURCE_DT, "return home"))
    return steps


def main():
    args = parse_args(__doc__)
    assets = fetch_assets(args.asset_cache)
    model = build_scene(assets, args.output / "scene.xml")
    robot_path = args.output / "robot.urdf"
    make_urdf(model, initial_state(model), robot_path)

    results = {}
    for enabled in [False, True] if args.compare else [args.datafarm]:
        name = "datafarm" if enabled else "baseline"
        results[name] = run(args, enabled, model, robot_path, plan_task=plan_task)

    (args.output / "comparison.json").write_text(json.dumps(results, indent=2))
    if args.compare and not args.no_render:
        compare_videos(
            args.output / "baseline/video.mp4",
            args.output / "datafarm/video.mp4",
            args.output / "comparison.mp4",
            fps=args.fps,
        )
    validate_results(results)


if __name__ == "__main__":
    main()
