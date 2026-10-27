"""CLI options, saved plans, and run orchestration for the pick-and-place demo."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from _pick_place_planning import SOURCE_DT, Planner
from _pick_place_simulation import simulate
from _pick_place_timing import make_timeline

from datafarm_pyroki import sample_plan, validate_trajectory
from datafarm_pyroki.assets import MODEL_ASSETS


def parse_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--datafarm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--compare",
        action="store_true",
        help="Run both modes and create a synchronized split-screen video",
    )
    p.add_argument("--output", type=Path, default=Path("work/pick-place-comparison"))
    p.add_argument("--asset-cache", type=Path, default=Path("work/menagerie-demo"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--encoder-weight", type=float, default=0.3)
    p.add_argument(
        "--retime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable learned trajectory segment retiming with DATAFARM (default: enabled)",
    )
    p.add_argument("--retime-iterations", type=int, default=250)
    p.add_argument("--retime-starts", type=int, default=8)
    p.add_argument(
        "--reuse-plan", action="store_true", help="Render/simulate previously saved plans"
    )
    p.add_argument(
        "--no-render",
        action="store_true",
        help="Run contact physics and metrics without an OpenGL context",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()
    if args.fps <= 0 or args.width < 320 or args.height < 240 or args.width % 2 or args.height % 2:
        p.error("Use positive fps and even dimensions of at least 320x240")
    if args.retime_starts < 1 or args.retime_iterations < 1:
        p.error("Retiming starts and iterations must be positive")
    if not np.isfinite(args.encoder_weight) or args.encoder_weight < 0:
        p.error("Encoder weight must be finite and nonnegative")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    return args


def run(args, enabled, model, robot_path, *, plan_task):
    retimed = enabled and args.retime
    name = "datafarm" if enabled else "baseline"
    output = args.output / name
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.npz"
    metadata_path = output / "planning.json"
    if args.reuse_plan and plan_path.exists():
        with np.load(plan_path, allow_pickle=False) as z:
            q, times = z["positions"], z["times"]
        metadata = json.loads(metadata_path.read_text())
        if "free_space_constraints" not in metadata or "trajectory_segments" not in metadata:
            raise ValueError("Saved plan uses an older metadata format; rerun without --reuse-plan")
        if bool(metadata.get("retiming_enabled", bool(metadata["trajectory_segments"]))) != retimed:
            raise ValueError(
                "Saved plan uses different retiming settings; rerun without --reuse-plan"
            )
        events = metadata["events"]
    else:
        planner = Planner(
            model, robot_path, datafarm=enabled, seed=args.seed, encoder_weight=args.encoder_weight
        )
        before = time.monotonic()
        steps = plan_task(planner)
        source = sample_plan(steps, ease_boundaries=False)
        source_events = [{"time": e.time, "command": e.event.command} for e in source.events]
        np.savez_compressed(
            output / "source_plan.npz", positions=source.positions, times=source.times
        )
        q, times, events, trajectory_segments, retime_seconds = make_timeline(planner, steps, args)
        metadata = {
            "datafarm": enabled,
            "retiming_enabled": retimed,
            "seed": args.seed,
            "encoder_weight": args.encoder_weight if enabled else 0.0,
            "planning_wall_seconds": time.monotonic() - before,
            "retiming_wall_seconds": retime_seconds,
            "source_dt": SOURCE_DT,
            "free_space_constraints": "joint endpoints, joint/velocity limits and table clearance",
            "source_events": source_events,
            "baseline_clock": {"boundary_ramp_fraction": 0.1, "learned_timing": False}
            if not retimed
            else None,
            "retiming_settings": {
                "clock_knots": 16,
                "starts": args.retime_starts,
                "iterations": args.retime_iterations,
                "velocity_cap_rad_s": 2.0,
                "acceleration_cap_rad_s2": 20.0,
            }
            if retimed
            else None,
            "mujoco_version": mujoco.__version__,
            "model_revisions": {
                name: {"repo_id": spec.repo_id, "revision": spec.revision, "sha256": spec.sha256}
                for name, spec in MODEL_ASSETS.items()
            },
            "events": events,
            "ik": planner.ik_records,
            "legs": planner.leg_records,
            "trajectory_segments": trajectory_segments,
        }
        np.savez_compressed(plan_path, positions=q, times=times)
        metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Simulating {name}: {times[-1]:.2f}s of arm/gripper commands", flush=True)
    ranges = np.array([model.joint(f"joint{i}").range for i in range(1, 8)])
    validate_trajectory(
        q,
        times,
        lower_limits=ranges[:, 0],
        upper_limits=ranges[:, 1],
        velocity_limits=np.full(7, 2.0),
        acceleration_limits=np.full(7, 20.0),
        position_tolerance=0.002,
    )
    metrics = simulate(
        model,
        q,
        times,
        events,
        enabled=enabled,
        retimed=retimed,
        output=output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        render=not args.no_render,
    )
    metrics["mean_selected_posture_nll"] = float(
        np.mean([record["selected_nll"] for record in metadata["ik"]])
    )
    metrics["mean_leg_style_cost"] = float(
        np.mean([record["style_cost"] for record in metadata["legs"]])
    )
    metrics["retimed_trajectory_segments"] = len(metadata["trajectory_segments"])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def validate_results(results):
    """Require successful physical grasps, collision checks, and all placements."""
    if any(
        not all(
            check["lifted"] and check["bilateral_contact_steps"] > 0
            for check in value["grasp_checks"]
        )
        for value in results.values()
    ):
        raise RuntimeError("A physical grasp/lift check failed; inspect the saved histories")
    if any(value["unintended_robot_contact_steps"] > 0 for value in results.values()):
        raise RuntimeError("Unintended robot contacts were detected; inspect saved diagnostics")
    if any(value["cubes_on_disc"] != 3 for value in results.values()):
        raise RuntimeError("A run did not place all three cubes; inspect the saved metrics/video")
