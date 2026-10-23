"""Physically pick/place three cubes with a DROID-style Franka in MuJoCo.

uv run --extra planning --extra simulation python examples/pick_place.py --compare
Use --datafarm / --no-datafarm for a single run. All downloaded meshes, plans and
videos default to the ignored work/ directory. No robot hardware is controlled.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from _pick_place_planning import SOURCE_DT, Planner
from _pick_place_scene import (
    DISC_CENTER,
    DISC_RADIUS,
    DISC_TOP,
    HOME,
    STARTS,
    TARGETS,
    build_scene,
    fetch_assets,
    make_urdf,
)
from PIL import Image, ImageDraw, ImageFont

from datafarm_pyroki import GripperEvent, MotionLeg, RetimeConfig, RetimedStroke, retime_plan
from datafarm_pyroki.assets import MODEL_ASSETS


def initial_state(model):
    data = mujoco.MjData(model)
    for i, q in enumerate(HOME, 1):
        data.qpos[model.joint(f"joint{i}").qposadr] = q
    data.ctrl[:7] = HOME
    data.ctrl[7] = 0.0  # Robotiq: zero=open, 255=close
    mujoco.mj_forward(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)
    data.time = 0.0
    return data


def baseline_timeline(steps):
    """Use the same fixed waypoint clock, without DataFarm's stroke retimer."""
    positions, times, events = [HOME.copy()], [0.0], []
    elapsed = 0.0
    for step in steps:
        if isinstance(step, GripperEvent):
            events.append({"time": elapsed, "command": step.command})
            count = max(1, int(np.ceil(step.dwell_seconds / 0.02)))
            times.extend(elapsed + np.linspace(0, step.dwell_seconds, count + 1)[1:])
            positions.extend([positions[-1].copy() for _ in range(count)])
            elapsed += step.dwell_seconds
        else:
            times.extend(elapsed + np.arange(1, len(step.positions)) * step.dt)
            positions.extend(step.positions[1:])
            elapsed = times[-1]
    return np.asarray(positions), np.asarray(times), events


def plan_task(planner):
    steps = []
    current = HOME.copy()
    for cube, (start, target) in enumerate(zip(STARTS, TARGETS, strict=True)):
        print(f"Planning cube {cube + 1}/3", flush=True)
        goals = [
            (np.r_[start[:2], 0.22], "approach"),
            (start, "descend to grasp"),
            (np.r_[start[:2], 0.22], "lift"),
            (np.r_[target[:2], 0.22], "transport"),
            (target, "lower onto disc"),
            (np.r_[target[:2], 0.22], "retreat"),
        ]
        for stage, (xyz, name) in enumerate(goals):
            goal = planner.endpoint(xyz, current, allowed_cube=cube)
            label = f"cube {cube + 1}: {name}"
            path = planner.leg(current, goal, label, allowed_cube=cube)
            steps.append(MotionLeg(path, SOURCE_DT, label))
            current = goal
            if stage == 1:
                steps.append(
                    GripperEvent(
                        {"action": "close", "cube": cube}, gripper="robotiq", dwell_seconds=0.7
                    )
                )
            if stage == 4:
                steps.append(
                    GripperEvent(
                        {"action": "open", "cube": cube}, gripper="robotiq", dwell_seconds=0.55
                    )
                )
        # Future collision checks see cubes already placed on the disc.
        address = planner.model.joint(f"cube{cube}_free").qposadr[0]
        planner.scratch.qpos[address : address + 3] = target
    steps.append(MotionLeg(planner.leg(current, HOME, "return home"), SOURCE_DT, "return home"))
    return steps


def make_timeline(planner, steps, args):
    before = time.monotonic()
    if not planner.datafarm:
        q, times, events = baseline_timeline(steps)
        return q, times, events, [], 0.0
    print("Retiming gripper-delimited strokes with the frozen encoder…", flush=True)
    config = RetimeConfig(
        knots=32,
        starts=args.retime_starts,
        iterations=args.retime_iterations,
        min_duration_multiplier=0.7,
        max_duration_multiplier=3.0,
    )
    plan = retime_plan(
        planner.encoder,
        steps,
        joint_map=planner.mapping,
        output_dt=0.02,
        velocity_limits=np.full(7, 2.0),
        acceleration_limits=np.full(7, 20.0),
        boundary_speed=0.0,
        config=config,
    )
    strokes = [step for step in plan.steps if isinstance(step, RetimedStroke)]
    records = [
        {"duration": s.duration, "style_before": s.original_style_cost, "style_after": s.style_cost}
        for s in strokes
    ]
    events = [{"time": event.time, "command": event.event.command} for event in plan.events]
    return plan.positions, plan.times, events, records, time.monotonic() - before


def _font(size):
    for path in (
        "/System/Library/Fonts/SFNS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def annotate(frame, *, enabled, clock, duration, phase, placed, width, height):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, width, 78), fill=(10, 17, 29, 235))
    accent = (75, 221, 179) if enabled else (104, 165, 255)
    draw.rounded_rectangle((22, 20, 31, 58), radius=4, fill=(*accent, 255))
    draw.text(
        (46, 13),
        "DATAFARM ON" if enabled else "BASELINE · DATAFARM OFF",
        font=_font(25),
        fill="white",
    )
    detail = (
        "Posture NLL + motion cost + stroke retiming"
        if enabled
        else "Nearest feasible IK + PyRoki fixed-clock trajopt"
    )
    draw.text((46, 45), detail, font=_font(15), fill=(174, 190, 210))
    draw.rectangle((0, height - 66, width, height), fill=(10, 17, 29, 235))
    draw.text((24, height - 56), phase.upper(), font=_font(19), fill="white")
    draw.text(
        (24, height - 29),
        f"{clock:05.1f}s / {duration:.1f}s   •   ON DISC: {placed}/3   •   MuJoCo contact dynamics",
        font=_font(14),
        fill=(174, 190, 210),
    )
    draw.rectangle(
        (0, height - 4, int(width * min(clock / duration, 1)), height), fill=(*accent, 255)
    )
    return np.asarray(image)


def placed_mask(positions):
    return (np.linalg.norm(positions[:, :2] - DISC_CENTER, axis=1) < DISC_RADIUS - 0.025) & (
        np.abs(positions[:, 2] - (DISC_TOP + 0.02)) < 0.015
    )


class VideoWriter:
    """Stream RGB frames to ffmpeg using posix_spawn, safe after JAX starts threads."""

    def __init__(self, path, width, height, fps):
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            close_fds=False,
        )

    def send(self, frame):
        self.process.stdin.write(frame.tobytes())

    def close(self):
        self.process.stdin.close()
        errors = self.process.stderr.read().decode()
        code = self.process.wait()
        self.process.stderr.close()
        if code:
            raise RuntimeError(f"ffmpeg failed: {errors}")


def simulate(
    model, q, times, events, *, enabled, output, fps=30, width=960, height=720, render=True
):
    data = initial_state(model)
    ids = [model.body(f"cube{i}").id for i in range(3)]
    arm_addresses = [model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)]
    duration = float(times[-1]) + 1.2
    renderer = mujoco.Renderer(model, height=height, width=width) if render else None
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.36, 0.015, 0.30]
    camera.distance, camera.azimuth, camera.elevation = 1.8, 135, -30
    writer = None
    if render:
        writer = VideoWriter(output / "video.mp4", width, height, fps)
    next_frame, event_index = 0.0, 0
    grip, phase = 0.0, "approaching first cube"
    log_times, actual_q, commanded_q, cubes, grasp_checks = [], [], [], [], []
    held_cube = None
    peak_heights = np.zeros(3)
    bilateral_contacts = np.zeros(3, dtype=int)
    next_log = 0.0
    unintended_contacts = 0
    worst_penetration = 0.0
    try:
        while data.time < duration:
            t = float(data.time)
            while event_index < len(events) and t >= events[event_index]["time"]:
                event = events[event_index]
                command = event["command"]
                grip = 255.0 if command["action"] == "close" else 0.0
                phase = f"cube {command['cube'] + 1}: " + (
                    "grasp and transport" if grip else "release on disc"
                )
                held_cube = command["cube"] if grip else None
                event_index += 1
            desired = np.array([np.interp(min(t, times[-1]), times, q[:, j]) for j in range(7)])
            data.ctrl[:7] = desired
            data.ctrl[7] = grip
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or np.max(np.abs(data.qvel)) > 150:
                raise RuntimeError("Unstable MuJoCo simulation")
            for contact in data.contact:
                if contact.dist >= -0.001:
                    continue
                names = [
                    model.body(int(model.geom_bodyid[g])).name
                    for g in (contact.geom1, contact.geom2)
                ]
                robot_flags = [name.startswith(("link", "rq_", "attachment")) for name in names]
                if (
                    not any(robot_flags)
                    or ("link0" in names and "world" in names)
                    or all(name.startswith("rq_") for name in names)
                ):
                    continue
                if any(name.startswith("cube") for name in names) and any(
                    name.startswith("rq_") for name in names
                ):
                    continue  # Intended finger/object manipulation contact.
                unintended_contacts += 1
                worst_penetration = max(worst_penetration, -float(contact.dist))
            if t >= times[-1]:
                phase = "complete" if placed_mask(data.xpos[ids]).all() else "placement check"
            if held_cube is not None:
                peak_heights[held_cube] = max(peak_heights[held_cube], data.xpos[ids[held_cube], 2])
                fingers = set()
                for contact in data.contact:
                    if contact.dist > 0:
                        continue
                    bodies = [
                        model.body(int(model.geom_bodyid[g])).name
                        for g in (contact.geom1, contact.geom2)
                    ]
                    if f"cube{held_cube}" in bodies:
                        for name in bodies:
                            if name.startswith("rq_left"):
                                fingers.add("left")
                            if name.startswith("rq_right"):
                                fingers.add("right")
                if len(fingers) == 2:
                    bilateral_contacts[held_cube] += 1
            if data.time >= next_log:
                log_times.append(float(data.time))
                actual_q.append(data.qpos[arm_addresses].copy())
                commanded_q.append(desired)
                cubes.append(data.xpos[ids].copy())
                next_log += 0.02
            if render and data.time >= next_frame:
                renderer.update_scene(data, camera)
                frame = annotate(
                    renderer.render(),
                    enabled=enabled,
                    clock=t,
                    duration=duration,
                    phase=phase,
                    placed=int(placed_mask(data.xpos[ids]).sum()),
                    width=width,
                    height=height,
                )
                writer.send(frame)
                if next_frame == 0:
                    Image.fromarray(frame).save(output / "first_frame.png")
                next_frame += 1 / fps
        if render:
            renderer.update_scene(data, camera)
            Image.fromarray(
                annotate(
                    renderer.render(),
                    enabled=enabled,
                    clock=duration,
                    duration=duration,
                    phase="complete",
                    placed=int(placed_mask(data.xpos[ids]).sum()),
                    width=width,
                    height=height,
                )
            ).save(output / "last_frame.png")
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    grasp_checks = [
        {
            "cube": cube,
            "peak_height_while_grasped_m": float(peak_heights[cube]),
            "lifted": bool(peak_heights[cube] > 0.12),
            "bilateral_contact_steps": int(bilateral_contacts[cube]),
        }
        for cube in range(3)
    ]
    positions = data.xpos[ids].copy()
    success = placed_mask(positions)
    history = np.asarray(actual_q)
    tracking = history - np.asarray(commanded_q)
    np.savez_compressed(
        output / "simulation.npz",
        times=log_times,
        actual_positions=history,
        commanded_positions=commanded_q,
        cubes=cubes,
    )
    return {
        "duration_s": duration,
        "unintended_robot_contact_steps": unintended_contacts,
        "worst_unintended_penetration_m": worst_penetration,
        "cubes_on_disc": int(success.sum()),
        "cube_final_positions_m": positions.tolist(),
        "placement_xy_error_m": np.linalg.norm(positions[:, :2] - TARGETS[:, :2], axis=1).tolist(),
        "grasp_checks": grasp_checks,
        "tracking_rmse_rad": float(np.sqrt(np.mean(tracking**2))),
        "max_tracking_error_rad": float(np.abs(tracking).max()),
        "actual_joint_path_length_rad": float(
            np.linalg.norm(np.diff(history, axis=0), axis=1).sum()
        ),
        "actual_velocity_rms_rad_s": float(
            np.sqrt(np.mean(np.gradient(history, 0.02, axis=0) ** 2))
        ),
    }


def run(args, enabled, model, robot_path):
    name = "datafarm" if enabled else "baseline"
    output = args.output / name
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.npz"
    metadata_path = output / "planning.json"
    if args.reuse_plan and plan_path.exists():
        with np.load(plan_path, allow_pickle=False) as z:
            q, times = z["positions"], z["times"]
        metadata = json.loads(metadata_path.read_text())
        events = metadata["events"]
    else:
        planner = Planner(
            model, robot_path, datafarm=enabled, seed=args.seed, encoder_weight=args.encoder_weight
        )
        before = time.monotonic()
        steps = plan_task(planner)
        q, times, events, strokes, retime_seconds = make_timeline(planner, steps, args)
        metadata = {
            "datafarm": enabled,
            "seed": args.seed,
            "encoder_weight": args.encoder_weight if enabled else 0.0,
            "planning_wall_seconds": time.monotonic() - before,
            "retiming_wall_seconds": retime_seconds,
            "source_dt": SOURCE_DT,
            "retiming_settings": {
                "clock_knots": 32,
                "starts": args.retime_starts,
                "iterations": args.retime_iterations,
                "velocity_cap_rad_s": 2.0,
                "acceleration_cap_rad_s2": 20.0,
            }
            if enabled
            else None,
            "mujoco_version": mujoco.__version__,
            "model_revisions": {
                name: {"repo_id": spec.repo_id, "revision": spec.revision, "sha256": spec.sha256}
                for name, spec in MODEL_ASSETS.items()
            },
            "events": events,
            "ik": planner.ik_records,
            "legs": planner.leg_records,
            "strokes": strokes,
        }
        np.savez_compressed(plan_path, positions=q, times=times)
        metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Simulating {name}: {times[-1]:.2f}s of arm/gripper commands", flush=True)
    metrics = simulate(
        model,
        q,
        times,
        events,
        enabled=enabled,
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
    metrics["retimed_strokes"] = len(metadata["strokes"])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
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
    assets = fetch_assets(args.asset_cache)
    model = build_scene(assets, args.output / "scene.xml")
    data = initial_state(model)
    robot_path = args.output / "robot.urdf"
    make_urdf(model, data, robot_path)
    results = {}
    for enabled in [False, True] if args.compare else [args.datafarm]:
        results["datafarm" if enabled else "baseline"] = run(args, enabled, model, robot_path)
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2))
    if args.compare and not args.no_render:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        longest = max(value["duration_s"] for value in results.values())
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(args.output / "baseline/video.mp4"),
                "-i",
                str(args.output / "datafarm/video.mp4"),
                "-filter_complex",
                "[0:v]tpad=stop_mode=clone:stop_duration=120[left];[1:v]tpad=stop_mode=clone:stop_duration=120[right];[left][right]hstack=inputs=2[v]",
                "-map",
                "[v]",
                "-t",
                str(longest),
                "-c:v",
                "libx264",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                str(args.output / "comparison.mp4"),
            ],
            check=True,
            close_fds=False,
        )
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


if __name__ == "__main__":
    main()
