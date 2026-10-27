"""MuJoCo replay, diagnostics, and video output for the pick-and-place demo."""

from pathlib import Path

import mujoco
import numpy as np
from _pick_place_scene import DISC_CENTER, DISC_RADIUS, DISC_TOP, TARGETS, initial_state
from _simulation import replay_trajectory
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    for path in (
        "/System/Library/Fonts/SFNS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def annotate(frame, *, enabled, retimed, clock, duration, phase, placed, width, height):
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
        "Posture NLL + motion cost"
        + (" + trajectory segment retiming" if retimed else " · fixed trajectory segment clock")
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


def simulate(
    model,
    q,
    times,
    events,
    *,
    enabled,
    output,
    retimed=False,
    fps=30,
    width=960,
    height=720,
    render=True,
):
    data = initial_state(model)
    ids = [model.body(f"cube{i}").id for i in range(3)]
    duration = float(times[-1]) + 1.2
    phase = "approaching first cube"
    held_cube = None
    peak_heights = np.zeros(3)
    bilateral_contacts = np.zeros(3, dtype=int)
    unintended_contacts = 0
    worst_penetration = 0.0

    def on_event(command, data):
        nonlocal held_cube, phase
        grip = 255.0 if command["action"] == "close" else 0.0
        phase = f"cube {command['cube'] + 1}: " + (
            "grasp and transport" if grip else "release on disc"
        )
        held_cube = command["cube"] if grip else None
        data.ctrl[7] = grip

    def on_step(t, data):
        nonlocal unintended_contacts, worst_penetration, phase
        if np.max(np.abs(data.qvel)) > 150:
            raise RuntimeError("Unstable MuJoCo simulation")
        for contact in data.contact:
            if contact.dist >= -0.001:
                continue
            names = [
                model.body(int(model.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2)
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

    def frame_transform(frame, t, data):
        return annotate(
            frame,
            enabled=enabled,
            retimed=retimed,
            clock=t,
            duration=duration,
            phase=phase,
            placed=int(placed_mask(data.xpos[ids]).sum()),
            width=width,
            height=height,
        )

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.36, 0.015, 0.30]
    camera.distance, camera.azimuth, camera.elevation = 1.8, 135, -30
    replay = replay_trajectory(
        model,
        data,
        q,
        times,
        joint_names=[f"joint{i}" for i in range(1, 8)],
        actuator_ids=range(7),
        body_names=[f"cube{i}" for i in range(3)],
        output=output,
        events=[(e["time"], e["command"]) for e in events],
        on_event=on_event,
        on_step=on_step,
        frame_transform=frame_transform,
        camera=camera,
        fps=fps,
        width=width,
        height=height,
        render=render,
    )
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
    history = replay.actual_positions
    tracking = history - replay.commanded_positions
    command_velocity = np.gradient(q, times, axis=0)
    command_acceleration = np.gradient(command_velocity, times, axis=0)
    command_jerk = np.gradient(command_acceleration, times, axis=0)
    np.savez_compressed(
        output / "simulation.npz",
        times=replay.times,
        actual_positions=history,
        commanded_positions=replay.commanded_positions,
        cubes=replay.body_positions,
    )
    return {
        "duration_s": duration,
        "commanded_acceleration_rms_rad_s2": float(np.sqrt(np.mean(command_acceleration**2))),
        "commanded_jerk_rms_rad_s3": float(np.sqrt(np.mean(command_jerk**2))),
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
