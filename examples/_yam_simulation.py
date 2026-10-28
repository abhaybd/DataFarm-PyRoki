"""Physical two-arm playback, manipulation checks, and comparison-video annotations."""

import mujoco
import numpy as np
from _pick_place_simulation import _font
from _simulation import replay_trajectory
from _yam_planning import robot_contact
from _yam_scene import ARMS, JOINT_NAMES, TARGETS, TRAY_TOP, initial_state
from PIL import Image, ImageDraw


def placed_mask(positions):
    return (np.linalg.norm(positions[:, :2] - TARGETS[:, :2], axis=1) < 0.035) & (
        np.abs(positions[:, 2] - (TRAY_TOP + 0.02)) < 0.012
    )


def annotate(frame, t, data, *, enabled, retimed, duration, placed, width, height):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    accent = (77, 225, 181) if enabled else (108, 170, 255)
    draw.rectangle((0, 0, width, 92), fill=(10, 17, 29, 245))
    draw.rounded_rectangle((20, 20, 28, 69), radius=4, fill=(*accent, 255))
    draw.text(
        (42, 12),
        "DATAFARM ON" if enabled else "BASELINE · DATAFARM OFF",
        font=_font(25),
        fill="white",
    )
    detail = (
        ("Per-arm posture + motion" + (" + learned timing" if retimed else " · fixed timing"))
        if enabled
        else "Nearest feasible IK · smoothness trajopt · fixed timing"
    )
    draw.text((42, 45), detail, font=_font(15), fill=(180, 195, 215))
    draw.text(
        (42, 66),
        "BIMANUAL YAM  /  6 DoF per arm  /  independent 30 Hz models",
        font=_font(13),
        fill=(145, 164, 187),
    )
    draw.rectangle((0, height - 65, width, height), fill=(10, 17, 29, 245))
    draw.text((22, height - 55), f"CUBES ON TRAY  {placed}/4", font=_font(20), fill="white")
    draw.text(
        (22, height - 28),
        f"{t:05.1f}s / {duration:.1f}s   ·   normal speed   ·   MuJoCo contact dynamics",
        font=_font(14),
        fill=(180, 195, 215),
    )
    draw.rectangle((0, height - 4, int(width * min(t / duration, 1)), height), fill=(*accent, 255))
    return np.asarray(image)


def simulate(
    model, q, times, events, output, *, enabled, retimed, fps=30, width=960, height=720, render=True
):
    data = initial_state(model)
    cube_bodies = [model.body(f"cube{i}").id for i in range(4)]
    held = {a: None for a in ARMS}
    peaks = np.zeros(4)
    contacts = np.zeros(4, dtype=int)
    bad = 0
    penetration = 0.0
    bad_examples = []
    duration = float(times[-1]) + 1.2

    def event(command, data):
        arm = command["arm"]
        closed = command["action"] == "close"
        data.ctrl[model.actuator(arm + "_gripper").id] = -0.005 if closed else 0.0475
        held[arm] = command["cube"] if closed else None

    def observe(t, data):
        nonlocal bad, penetration
        for c in data.contact:
            if c.dist >= -0.0015:
                continue
            names = [model.body(int(model.geom_bodyid[g])).name for g in (c.geom1, c.geom2)]
            cube = next((int(n[4:]) for n in names if n.startswith("cube")), None)
            if robot_contact(model, c, cube):
                bad += 1
                penetration = max(penetration, -float(c.dist))
                if len(bad_examples) < 10:
                    bad_examples.append(dict(time=t, bodies=names, depth=float(-c.dist)))
        for arm, cube in held.items():
            if cube is None:
                continue
            peaks[cube] = max(peaks[cube], data.xpos[cube_bodies[cube], 2])
            sides = set()
            for c in data.contact:
                if c.dist > 0:
                    continue
                names = [model.body(int(model.geom_bodyid[g])).name for g in (c.geom1, c.geom2)]
                if f"cube{cube}" not in names:
                    continue
                for side in ("left", "right"):
                    if any(n.startswith(f"{arm}_{side}_") for n in names):
                        sides.add(side)
            if len(sides) == 2:
                contacts[cube] += 1

    def transform(frame, t, data):
        return annotate(
            frame,
            t,
            data,
            enabled=enabled,
            retimed=retimed,
            duration=duration,
            placed=int(placed_mask(data.xpos[cube_bodies]).sum()),
            width=width,
            height=height,
        )

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.24, 0, 0.19]
    camera.distance, camera.azimuth, camera.elevation = 1.5, 180, -38
    result = replay_trajectory(
        model,
        data,
        q,
        times,
        joint_names=JOINT_NAMES,
        actuator_ids=[model.actuator(n).id for n in JOINT_NAMES],
        output=output,
        events=[(e["time"], e["command"]) for e in events],
        on_event=event,
        on_step=observe,
        body_names=[f"cube{i}" for i in range(4)],
        frame_transform=transform,
        camera=camera,
        fps=fps,
        width=width,
        height=height,
        render=render,
    )
    final = data.xpos[cube_bodies].copy()
    error = result.actual_positions - result.commanded_positions
    velocity = np.gradient(q, times, axis=0)
    acceleration = np.gradient(velocity, times, axis=0)
    jerk = np.gradient(acceleration, times, axis=0)
    np.savez_compressed(
        output / "simulation.npz",
        times=result.times,
        actual_positions=result.actual_positions,
        commanded_positions=result.commanded_positions,
        cubes=result.body_positions,
    )
    return dict(
        duration_s=result.duration,
        cubes_on_tray=int(placed_mask(final).sum()),
        final_cube_positions_m=final.tolist(),
        placement_xy_error_m=np.linalg.norm(final[:, :2] - TARGETS[:, :2], axis=1).tolist(),
        grasp_checks=[
            dict(
                cube=i,
                lifted=bool(peaks[i] > 0.09),
                peak_height_m=float(peaks[i]),
                bilateral_contact_steps=int(contacts[i]),
            )
            for i in range(4)
        ],
        unintended_robot_contact_steps=bad,
        worst_unintended_penetration_m=penetration,
        contact_examples=bad_examples,
        tracking_rmse_rad=float(np.sqrt(np.mean(error**2))),
        max_tracking_error_rad=float(np.abs(error).max()),
        commanded_acceleration_rms_rad_s2=float(np.sqrt(np.mean(acceleration**2))),
        commanded_jerk_rms_rad_s3=float(np.sqrt(np.mean(jerk**2))),
    )
