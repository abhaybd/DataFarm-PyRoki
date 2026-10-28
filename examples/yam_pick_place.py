"""Compare learned left/right YAM marginals in a physical two-arm pick-and-place scene.

Run with planning and simulation extras:
MUJOCO_GL=egl python examples/yam_pick_place.py --compare
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from _simulation import compare_videos
from _yam_planning import Planner, plan_arm
from _yam_scene import ARMS, JOINT_NAMES, build_scene, fetch_assets, initial_state, make_urdf
from _yam_scene import TARGETS as INDEPENDENT_TARGETS
from _yam_simulation import simulate

from datafarm_pyroki import (
    RetimeConfig,
    RetimedTrajectorySegment,
    interpolate_plan,
    retime_plan,
    sample_plan,
    validate_trajectory,
)


def arm_timeline(planner, steps, args):
    if not planner.enabled or not args.retime:
        timeline = sample_plan(steps)
        return timeline, []
    config = RetimeConfig(
        knots=16,
        starts=args.retime_starts,
        iterations=args.retime_iterations,
        min_duration_multiplier=0.8,
        max_duration_multiplier=2.0,
    )
    timeline = retime_plan(
        planner.encoder,
        interpolate_plan(steps),
        joint_map=planner.mapping,
        output_dt=0.02,
        velocity_limits=np.full(6, 2.0),
        acceleration_limits=np.full(6, 20.0),
        boundary_speed=0.0,
        config=config,
    )
    records = [
        dict(
            duration=s.duration,
            style_before=s.original_style_cost,
            style_after=s.style_cost,
            settings=asdict(config),
        )
        for s in timeline.steps
        if isinstance(s, RetimedTrajectorySegment)
    ]
    return timeline, records


def combine(timelines):
    duration = max(float(t.times[-1]) for t in timelines)
    times = np.linspace(0, duration, int(np.ceil(duration / 0.02)) + 1)
    q = np.concatenate(
        [
            np.stack([np.interp(times, t.times, c) for c in t.positions.T], axis=1)
            for t in timelines
        ],
        axis=1,
    )
    events = sorted(
        [dict(time=float(e.time), command=e.event.command) for t in timelines for e in t.events],
        key=lambda e: e["time"],
    )
    return q, times, events


def run(args, model, enabled):
    name = "datafarm" if enabled else "baseline"
    output = args.output / name
    output.mkdir(exist_ok=True)
    metadata_path = output / "planning.json"
    plan_path = output / "plan.npz"
    retimed = enabled and args.retime
    if args.handoff:
        from _yam_handoff import TARGETS, plan_handoffs
    else:
        TARGETS = INDEPENDENT_TARGETS
    hashes = {
        f"{arm}/{file}": hashlib.sha256((args.models / arm / file).read_bytes()).hexdigest()
        for arm in ARMS
        for file in ("encoder.npz", "posture_prior.npz")
    }
    identity = dict(
        version=1,
        task="handoff_v2" if args.handoff else "independent",
        scene_sha256=hashlib.sha256((args.output / "scene.xml").read_bytes()).hexdigest(),
        solver_iterations=180,
        seed=args.seed,
        encoder_weight=args.encoder_weight if enabled else 0.0,
        retimed=retimed,
        retime_starts=args.retime_starts,
        retime_iterations=args.retime_iterations,
        models=hashes,
    )
    if args.reuse_plan and plan_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["identity"] != identity:
            raise ValueError("Saved plan settings/models differ; regenerate")
        with np.load(plan_path, allow_pickle=False) as z:
            q, times = z["positions"], z["times"]
        events = metadata["events"]
    else:
        metadata = dict(identity=identity, arms={}, events=[])
        timelines = []
        planners = {}
        for arm in ARMS:
            print(f"Planning {name}/{arm}", flush=True)
            planner = Planner(
                model,
                args.output / f"{arm}.urdf",
                arm,
                args.models,
                enabled=enabled,
                seed=args.seed,
                weight=args.encoder_weight,
            )
            planners[arm] = planner
            if args.handoff:
                continue
            steps = plan_arm(planner)
            source = sample_plan(steps, ease_boundaries=False)
            np.savez_compressed(
                output / f"{arm}_source.npz", positions=source.positions, times=source.times
            )
            timeline, records = arm_timeline(planner, steps, args)
            timelines.append(timeline)
            metadata["arms"][arm] = dict(
                ik=planner.ik_records,
                legs=planner.leg_records,
                retiming=records,
                duration_s=float(timeline.times[-1]),
            )
        if args.handoff:
            q, times, events, records = plan_handoffs(planners, args, arm_timeline)
            metadata.update(records)
        else:
            q, times, events = combine(timelines)
        metadata["events"] = events
        np.savez_compressed(plan_path, positions=q, times=times)
        metadata_path.write_text(json.dumps(metadata, indent=2))
    ranges = np.array([model.joint(n).range for n in JOINT_NAMES])
    validate_trajectory(
        q,
        times,
        lower_limits=ranges[:, 0],
        upper_limits=ranges[:, 1],
        velocity_limits=np.full(12, 2.0),
        acceleration_limits=np.full(12, 20.0),
        position_tolerance=0.002,
    )
    print(f"Simulating {name}: {times[-1]:.2f} seconds", flush=True)
    metrics = simulate(
        model,
        q,
        times,
        events,
        output,
        enabled=enabled,
        retimed=retimed,
        fps=args.fps,
        width=args.width,
        height=args.height,
        render=not args.no_render,
        targets=TARGETS,
        handoff=args.handoff,
    )
    metrics["arms"] = {
        arm: dict(
            mean_posture_nll=float(np.mean([r["selected_nll"] for r in m["ik"]])),
            mean_leg_style_cost=float(np.mean([r["style_cost"] for r in m["legs"]])),
        )
        for arm, m in metadata["arms"].items()
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    if (
        metrics["cubes_on_tray"] != len(TARGETS)
        or metrics["unintended_robot_contact_steps"]
        or not all(
            c["lifted"] and c["bilateral_contact_steps"] > 0 for c in metrics["grasp_checks"]
        )
    ):
        raise RuntimeError(
            f"{name}: physical manipulation/collision checks failed; inspect metrics"
        )
    if args.handoff and not all(
        c["released_to_receiver"]
        and c["dual_bilateral_contact_steps"] > 0
        and c["receiver_support_after_release_steps"] > 20
        for c in metrics["handoff_checks"]
    ):
        raise RuntimeError("A physical handoff validation failed; inspect metrics")
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare", action="store_true")
    p.add_argument(
        "--handoff",
        action="store_true",
        help="Transfer three cubes from the left side to the right via a physical handoff",
    )
    p.add_argument("--datafarm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--retime", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reuse-plan", action="store_true")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--models", type=Path, default=Path("work/yam/models"))
    p.add_argument("--asset-cache", type=Path, default=Path("work/yam-menagerie"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--encoder-weight", type=float, default=0.1)
    p.add_argument("--retime-starts", type=int, default=4)
    p.add_argument("--retime-iterations", type=int, default=150)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()
    if args.fps <= 0 or min(args.width, args.height) < 320 or args.width % 2 or args.height % 2:
        p.error("Use positive fps and even image dimensions of at least 320 pixels")
    if args.retime_starts < 1 or args.retime_iterations < 1:
        p.error("Retiming starts and iterations must be positive")
    if not np.isfinite(args.encoder_weight) or args.encoder_weight < 0:
        p.error("Encoder weight must be finite and nonnegative")
    args.output = (
        args.output or Path("work/yam-handoff" if args.handoff else "work/yam-pick-place")
    ).resolve()
    args.models = args.models.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {}
    if args.handoff:
        from _yam_handoff import scene_settings

        settings = scene_settings()
    model = build_scene(fetch_assets(args.asset_cache), args.output / "scene.xml", **settings)
    for arm in ARMS:
        make_urdf(model, initial_state(model), arm, args.output / f"{arm}.urdf")
    results = {}
    for enabled in [False, True] if args.compare else [args.datafarm]:
        results["datafarm" if enabled else "baseline"] = run(args, model, enabled)
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2))
    if args.compare and not args.no_render:
        compare_videos(
            args.output / "baseline/video.mp4",
            args.output / "datafarm/video.mp4",
            args.output / "comparison.mp4",
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
