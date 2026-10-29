"""Compare DATAFARM off, independent motion encoders, and one bimanual motion encoder.

All variants use the same joint planner, scene constraints and task barriers.
Run: MUJOCO_GL=egl python examples/yam_bimanual.py --compare
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from _pick_place_simulation import _font
from _yam_bimanual import BimanualPlanner
from _yam_handoff import TARGETS, plan_handoffs, scene_settings
from _yam_planning import Planner
from _yam_scene import (
    ARMS,
    JOINT_NAMES,
    build_scene,
    fetch_assets,
    initial_state,
    make_bimanual_urdf,
    make_urdf,
)
from _yam_simulation import simulate
from PIL import Image, ImageDraw
from yam_pick_place import arm_timeline

from datafarm_pyroki import validate_trajectory

VARIANTS = ("off", "independent", "bimanual")


def stack_videos(paths, destination, *, fps=30, speedup=2):
    """Stream equal-speed panels; hold finished panels at their final frame."""
    with ExitStack() as stack:
        readers = [stack.enter_context(iio.imopen(p, "r", plugin="pyav")) for p in paths]
        video = stack.enter_context(iio.imopen(destination, "w", plugin="pyav"))
        video.init_video_stream("libx264", fps=fps, pixel_format="yuv420p")
        last = [None] * len(paths)
        for index, frames in enumerate(zip_longest(*(r.iter() for r in readers))):
            for i, frame in enumerate(frames):
                if frame is not None:
                    last[i] = frame
            if index % speedup:
                continue
            if any(frame is None for frame in last):
                raise ValueError("Every comparison input must contain frames")
            image = Image.fromarray(np.concatenate(last, axis=1))
            draw = ImageDraw.Draw(image)
            width, height = image.size
            draw.rectangle((0, height - 29, width, height - 5), fill=(10, 17, 29))
            label = f"{speedup}× speed"
            bounds = draw.textbbox((0, 0), label, font=_font(30))
            x = width - (bounds[2] - bounds[0]) - 28
            draw.rectangle((x - 12, 12, width - 16, 62), fill=(3, 7, 13))
            draw.text((x, 16), label, font=_font(30), fill="white")
            for i in range(len(paths)):
                draw.text(
                    (i * last[i].shape[1] + 22, height - 28),
                    f"{speedup}× playback",
                    font=_font(14),
                    fill=(180, 195, 215),
                )
            video.write_frame(np.asarray(image))


def run(args, model, variant, hashes):
    output = args.output / variant
    output.mkdir(exist_ok=True)
    identity = dict(
        version=1,
        task="direct_handoff_joint_solver",
        variant=variant,
        models=hashes,
        scene_sha256=hashlib.sha256((args.output / "scene.xml").read_bytes()).hexdigest(),
        solver_iterations=320,
        seed=args.seed,
        encoder_weight=args.encoder_weight,
        retimed=args.retime and variant != "off",
        retime_starts=args.retime_starts,
        retime_iterations=args.retime_iterations,
        independent_objective="mean of marginal Mahalanobis costs",
        collision_geometry="coupled_mujoco_signed_distance_v1",
    )
    metadata_path = output / "planning.json"
    if args.reuse_plan and (output / "plan.npz").exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["identity"] != identity:
            raise ValueError("Saved models/settings differ; regenerate the plan")
        with np.load(output / "plan.npz", allow_pickle=False) as z:
            q, times = z["positions"], z["times"]
        events = metadata["events"]
    else:
        joint = BimanualPlanner(
            model, args.output / "bimanual.urdf", args.models, variant, weight=args.encoder_weight
        )
        arms = {
            arm: Planner(
                model,
                args.output / f"{arm}.urdf",
                arm,
                args.models,
                enabled=variant != "off",
                seed=args.seed,
                weight=0.0,
                direct=True,
            )
            for arm in ARMS
        }
        q, times, events, metadata = plan_handoffs(arms, args, arm_timeline, joint_planner=joint)
        metadata.update(identity=identity, events=events, joint_legs=joint.records)
        np.savez_compressed(output / "plan.npz", positions=q, times=times)
        np.savez_compressed(
            output / "source_paths.npz", positions=np.stack(joint.source_paths), source_dt=0.1
        )
        metadata_path.write_text(json.dumps(metadata, indent=2))
    limits = np.array([model.joint(n).range for n in JOINT_NAMES])
    validate_trajectory(
        q,
        times,
        lower_limits=limits[:, 0],
        upper_limits=limits[:, 1],
        velocity_limits=np.full(12, 2.0),
        acceleration_limits=np.full(12, 20.0),
        position_tolerance=0.002,
    )
    print(f"[replay] {variant}: {times[-1]:.2f} seconds", flush=True)
    metrics = simulate(
        model,
        q,
        times,
        events,
        output,
        enabled=variant != "off",
        retimed=identity["retimed"],
        fps=args.fps,
        width=args.width,
        height=args.height,
        render=not args.no_render,
        targets=TARGETS,
        handoff=True,
        direct=True,
        variant=variant,
    )
    metrics["variant"] = variant
    metrics["joint_motion_cost_mean"] = float(
        np.mean([r["style_cost"] for r in metadata["joint_legs"]])
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if metrics["cubes_on_tray"] != len(TARGETS) or metrics["unintended_robot_contact_steps"]:
        raise RuntimeError(f"{variant}: placement/collision validation failed; inspect metrics")
    if not all(c["lifted"] and c["bilateral_contact_steps"] > 0 for c in metrics["grasp_checks"]):
        raise RuntimeError(f"{variant}: a physical grasp failed")
    if not all(
        h["released_to_receiver"]
        and h["dual_bilateral_contact_steps"] > 0
        and h["receiver_support_after_release_steps"] > 20
        for h in metrics["handoff_checks"]
    ):
        raise RuntimeError(f"{variant}: a physical handoff failed")
    print(
        f"[validated] {variant}: 3/3 handoffs and placements, zero unintended contacts", flush=True
    )
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--compare",
        action="store_true",
        help="Run all three conditions and compose both comparisons",
    )
    p.add_argument("--variant", choices=VARIANTS, default="bimanual")
    p.add_argument("--models", type=Path, default=Path("work/yam/models"))
    p.add_argument("--output", type=Path, default=Path("work/yam-bimanual-demo"))
    p.add_argument("--asset-cache", type=Path, default=Path("work/yam-menagerie"))
    p.add_argument("--reuse-plan", action="store_true")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--retime", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--encoder-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--retime-starts", type=int, default=4)
    p.add_argument("--retime-iterations", type=int, default=150)
    p.add_argument("--speedup", type=int, choices=(1, 2, 4), default=2)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()
    if args.fps <= 0 or min(args.width, args.height) < 320 or args.width % 2 or args.height % 2:
        p.error("Use positive fps and even dimensions of at least 320 pixels")
    if not np.isfinite(args.encoder_weight) or args.encoder_weight < 0:
        p.error("Encoder weight must be finite and nonnegative")
    if min(args.retime_starts, args.retime_iterations) < 1:
        p.error("Retiming budget must be positive")
    args.direct_transfers = True
    args.output = args.output.resolve()
    args.models = args.models.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {
        f"{arm}/{name}": hashlib.sha256((args.models / arm / name).read_bytes()).hexdigest()
        for arm, names in (
            ("left", ("encoder.npz", "posture_prior.npz")),
            ("right", ("encoder.npz", "posture_prior.npz")),
            ("bimanual", ("encoder.npz",)),
        )
        for name in names
    }
    model = build_scene(
        fetch_assets(args.asset_cache), args.output / "scene.xml", **scene_settings()
    )
    for arm in ARMS:
        make_urdf(model, initial_state(model), arm, args.output / f"{arm}.urdf")
    make_bimanual_urdf(args.output, args.output / "bimanual.urdf")
    for variant in VARIANTS if args.compare else (args.variant,):
        run(args, model, variant, hashes)
    summary = {
        v: json.loads((args.output / v / "metrics.json").read_text())
        for v in VARIANTS
        if (args.output / v / "metrics.json").exists()
    }
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2))
    if args.compare and not args.no_render:
        stack_videos(
            [args.output / v / "video.mp4" for v in ("off", "bimanual")],
            args.output / "off_vs_bimanual.mp4",
            fps=args.fps,
            speedup=args.speedup,
        )
        stack_videos(
            [args.output / v / "video.mp4" for v in VARIANTS],
            args.output / "all_three.mp4",
            fps=args.fps,
            speedup=args.speedup,
        )


if __name__ == "__main__":
    main()
