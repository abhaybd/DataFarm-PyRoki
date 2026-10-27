"""Shared MuJoCo trajectory playback and streaming video support for examples.

Scene initialization, gripper commands, contact policies, and success criteria
belong to the caller. This module knows only joint/actuator/body mappings and time.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from itertools import zip_longest

import imageio.v3 as iio
import mujoco
import numpy as np


@dataclass(frozen=True)
class ReplayResult:
    duration: float
    times: np.ndarray
    actual_positions: np.ndarray
    commanded_positions: np.ndarray
    body_positions: np.ndarray


def replay_trajectory(
    model,
    data,
    positions,
    times,
    *,
    joint_names,
    actuator_ids,
    output,
    events=(),
    on_event=None,
    on_step=None,
    body_names=(),
    frame_transform=None,
    camera=None,
    settle_seconds=1.2,
    log_dt=0.02,
    fps=30,
    width=960,
    height=720,
    render=True,
) -> ReplayResult:
    """Replay joint targets with optional timed commands and observation callbacks.

    Events are (local_time, arbitrary_command) pairs. on_event(command, data)
    handles grippers or other task actions; on_step(local_time, data) observes
    every physics step. Frame transforms receive (RGB, local_time, data).
    Logged times are relative to the supplied data's initial time. The caller's
    data is advanced in place; source trajectories and event payloads are retained.
    """
    q, times = np.asarray(positions), np.asarray(times)
    if (
        q.ndim != 2
        or len(q) < 2
        or q.shape[1] != len(joint_names)
        or times.shape != (len(q),)
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(q).all()
        or not np.isfinite(times).all()
    ):
        raise ValueError("Need finite joint positions and increasing local times starting at zero")
    actuator_ids = np.asarray(actuator_ids, dtype=int)
    if (
        actuator_ids.shape != (q.shape[1],)
        or np.any(actuator_ids < 0)
        or np.any(actuator_ids >= model.nu)
    ):
        raise ValueError("actuator_ids must map each trajectory joint to one model actuator")
    if len(np.unique(actuator_ids)) != len(actuator_ids):
        raise ValueError("actuator_ids must be distinct")
    if (
        not np.isfinite(settle_seconds)
        or settle_seconds < 0
        or not np.isfinite(log_dt)
        or log_dt <= 0
        or fps <= 0
    ):
        raise ValueError("Need nonnegative settling time and positive log_dt/fps")
    events = tuple(events)
    event_times = np.asarray([event[0] for event in events])
    if (
        not np.isfinite(event_times).all()
        or np.any(event_times < 0)
        or np.any(np.diff(event_times) < 0)
        or np.any(event_times > times[-1])
    ):
        raise ValueError("Events must be ordered within the source trajectory duration")
    if events and on_event is None:
        raise ValueError("Timed events require an on_event callback")
    joints = [model.joint(name) for name in joint_names]
    if any(
        int(np.asarray(j.type).item())
        not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        for j in joints
    ):
        raise ValueError("Playback joints must each have one position coordinate")
    addresses = [joint.qposadr[0] for joint in joints]
    bodies = [model.body(name).id for name in body_names]
    duration = float(times[-1]) + settle_seconds
    start_time = float(data.time)
    next_frame = next_log = 0.0
    event_index = 0
    log_times, actual, commanded, body_positions = [], [], [], []

    with ExitStack() as resources:
        renderer = writer = None
        if render:
            renderer = mujoco.Renderer(model, height=height, width=width)
            resources.callback(renderer.close)
            writer = resources.enter_context(iio.imopen(output / "video.mp4", "w", plugin="pyav"))
            writer.init_video_stream("libx264", fps=fps, pixel_format="yuv420p")

        def frame_at(t):
            if camera is None:
                renderer.update_scene(data)
            else:
                renderer.update_scene(data, camera)
            frame = renderer.render()
            return frame if frame_transform is None else frame_transform(frame, t, data)

        while data.time - start_time < duration:
            t = float(data.time - start_time)
            while event_index < len(events) and t >= events[event_index][0]:
                on_event(events[event_index][1], data)
                event_index += 1
            desired = np.array([np.interp(min(t, times[-1]), times, column) for column in q.T])
            data.ctrl[actuator_ids] = desired
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Nonfinite MuJoCo simulation state")
            if on_step is not None:
                on_step(t, data)
            if data.time - start_time >= next_log:
                log_times.append(float(data.time - start_time))
                actual.append(data.qpos[addresses].copy())
                commanded.append(desired)
                body_positions.append(data.xpos[bodies].copy())
                next_log += log_dt
            if render and data.time - start_time >= next_frame:
                frame = frame_at(t)
                writer.write_frame(frame)
                if next_frame == 0:
                    iio.imwrite(output / "first_frame.png", frame)
                next_frame += 1 / fps
        # An event at the final timestamp still fires when no settling is requested.
        while event_index < len(events):
            on_event(events[event_index][1], data)
            event_index += 1
        if render:
            iio.imwrite(output / "last_frame.png", frame_at(duration))
    return ReplayResult(
        duration,
        np.asarray(log_times),
        np.asarray(actual),
        np.asarray(commanded),
        np.asarray(body_positions),
    )


def compare_videos(left_path, right_path, output_path, *, fps):
    """Stream two videos side by side, holding the shorter final frame."""
    with (
        iio.imopen(left_path, "r", plugin="pyav") as left,
        iio.imopen(right_path, "r", plugin="pyav") as right,
        iio.imopen(output_path, "w", plugin="pyav") as video,
    ):
        video.init_video_stream("libx264", fps=fps, pixel_format="yuv420p")
        last_left = last_right = None
        for left_frame, right_frame in zip_longest(left.iter(), right.iter()):
            if left_frame is not None:
                last_left = left_frame
            if right_frame is not None:
                last_right = right_frame
            if last_left is None or last_right is None:
                raise ValueError("Both comparison videos must contain at least one frame")
            video.write_frame(np.concatenate([last_left, last_right], axis=1))
