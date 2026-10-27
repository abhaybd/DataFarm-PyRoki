"""Group motion legs at gripper events and retime/stitch a complete plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import numpy as np

from .embodiment import JointMap
from .encoder import MotionEncoder
from .retiming import (
    RetimeConfig,
    RetimedTrajectorySegment,
    RetimingError,
    _limits,
    retime_trajectory_segment,
)


@dataclass(frozen=True)
class MotionLeg:
    positions: np.ndarray  # [time,robot_joints]; accepts JAX arrays too
    dt: float
    label: str = ""


@dataclass(frozen=True)
class GripperEvent:
    command: object  # preserved verbatim for your executor
    gripper: str = "gripper"
    dwell_seconds: float = 0.0  # optional stationary hold, not optimized

    def __post_init__(self):
        if not np.isfinite(self.dwell_seconds) or self.dwell_seconds < 0:
            raise ValueError("Gripper dwell must be nonnegative and finite")


@dataclass(frozen=True)
class TrajectorySegment:
    """Consecutive motion legs joined between gripper events for retiming."""

    positions: np.ndarray
    times: np.ndarray
    labels: tuple[str, ...]

    @property
    def duration(self) -> float:
        return float(self.times[-1])


@dataclass(frozen=True)
class TimedGripperEvent:
    time: float
    event: GripperEvent


@dataclass(frozen=True)
class RetimedPlan:
    steps: tuple[RetimedTrajectorySegment | GripperEvent, ...]
    events: tuple[TimedGripperEvent, ...]
    positions: np.ndarray
    times: np.ndarray  # absolute plan times; segments may have slightly different dt
    velocities: np.ndarray
    accelerations: np.ndarray


def group_trajectory_segments(
    steps: Sequence[MotionLeg | GripperEvent],
    *,
    position_tolerance: float = 1e-6,
) -> tuple[TrajectorySegment | GripperEvent, ...]:
    """Join consecutive legs while retaining event identity, ordering and source timing.

    Every gripper event ends the current trajectory segment. No action/arm name assumptions
    are made. A whole-body model can therefore delimit trajectory segments using any gripper.
    Discontinuous leg endpoints are rejected, including across gripper events.
    """
    if not np.isfinite(position_tolerance) or position_tolerance < 0:
        raise ValueError("position_tolerance must be nonnegative and finite")
    out, positions, times, labels = [], [], [], []
    previous = None
    dof = None

    def flush():
        if positions:
            out.append(
                TrajectorySegment(np.concatenate(positions), np.concatenate(times), tuple(labels))
            )
            positions.clear()
            times.clear()
            labels.clear()

    for step in steps:
        if isinstance(step, GripperEvent):
            flush()
            out.append(step)
            continue
        if not isinstance(step, MotionLeg):
            raise TypeError("Plan steps must be MotionLeg or GripperEvent")
        q = np.asarray(step.positions, dtype=np.float64)
        if q.ndim != 2 or len(q) < 2 or not np.isfinite(q).all() or q.shape[1] == 0:
            raise ValueError("Each motion leg needs finite positions [time>=2,robot_joints]")
        if not np.isfinite(step.dt) or step.dt <= 0:
            raise ValueError("Each leg dt must be positive and finite")
        dof = q.shape[1] if dof is None else dof
        if q.shape[1] != dof:
            raise ValueError("All plan legs must use the same robot joint order and dimension")
        if previous is not None and not np.allclose(
            q[0], previous, atol=position_tolerance, rtol=0
        ):
            raise ValueError("Discontinuous motion endpoints, including across gripper events")
        offset = float(times[-1][-1]) if times else 0.0
        local = offset + np.arange(len(q)) * step.dt
        if positions:
            positions.append(q[1:])
            times.append(local[1:])
        else:
            positions.append(q)
            times.append(local)
        labels.append(step.label)
        previous = q[-1]
    flush()
    return tuple(out)


def _stitch_plan(steps, velocity_limits, acceleration_limits, *, output_dt):
    trajectory_segments = [step for step in steps if isinstance(step, RetimedTrajectorySegment)]
    if not trajectory_segments:
        raise ValueError("A retimed plan needs at least one trajectory segment")
    q_chunks = [trajectory_segments[0].positions[:1]]
    t_chunks = [np.array([0.0])]
    pose = trajectory_segments[0].positions[0]
    elapsed = 0.0
    events = []
    for step in steps:
        if isinstance(step, GripperEvent):
            events.append(TimedGripperEvent(elapsed, step))
            if step.dwell_seconds > 0:
                count = max(1, int(np.ceil(step.dwell_seconds / output_dt)))
                local = np.linspace(0.0, step.dwell_seconds, count + 1)[1:]
                t_chunks.append(elapsed + local)
                q_chunks.append(np.repeat(pose[None], count, axis=0))
                elapsed += step.dwell_seconds
        else:
            if not np.allclose(pose, step.positions[0], rtol=0, atol=1e-6):
                raise ValueError("Retimed trajectory segment endpoints are discontinuous")
            q_chunks.append(step.positions[1:])
            t_chunks.append(elapsed + step.times[1:])
            elapsed += step.duration
            pose = step.positions[-1]
    q = np.concatenate(q_chunks)
    t = np.concatenate(t_chunks)
    if np.any(np.diff(t) <= 0):
        raise ValueError("Stitched timestamps must be strictly increasing")
    # Recompute on the full signal, so a reversal/hold at a gripper boundary cannot
    # evade acceleration checks merely because each trajectory segment was feasible in isolation.
    velocity = np.gradient(q, t, axis=0)
    velocity[0], velocity[-1] = 0.0, 0.0
    acceleration = np.gradient(velocity, t, axis=0)
    interval_speed = np.diff(q, axis=0) / np.diff(t)[:, None]
    if (
        not np.isfinite(velocity).all()
        or not np.isfinite(acceleration).all()
        or np.any(np.abs(interval_speed) > velocity_limits)
        or np.any(np.abs(velocity) > velocity_limits)
        or np.any(np.abs(acceleration) > acceleration_limits)
    ):
        raise RetimingError(
            "Stitched motion exceeds limits, including at a trajectory segment/gripper junction; "
            "reduce boundary speed or change the duration/limit settings"
        )
    return RetimedPlan(tuple(steps), tuple(events), q, t, velocity, acceleration)


def retime_plan(
    encoder: MotionEncoder,
    steps: Sequence[MotionLeg | GripperEvent],
    *,
    joint_map: JointMap,
    output_dt: float,
    velocity_limits,
    acceleration_limits,
    boundary_speed: float = 0.07,
    config: RetimeConfig = RetimeConfig(),
    key: jax.Array | None = None,
    trajectory_validator: Callable[[np.ndarray, np.ndarray], bool] | None = None,
) -> RetimedPlan:
    """Retime trajectory segments, preserve gripper events, and validate the stitched plan.

    Boundary speed is a caller-selected soft target at gripper-adjacent ends.
    Only a trajectory segment at a plan's absolute beginning/end requests exact rest there.
    A provided PRNG key samples an independent target per trajectory segment; default is the
    reference mean. Callback timestamps are LOCAL for each candidate, then absolute
    for the final stitched validation. The function never executes any commands.
    """
    if not np.isfinite(boundary_speed) or boundary_speed < 0:
        raise ValueError("boundary_speed must be nonnegative and finite")
    if not np.isfinite(output_dt) or output_dt <= 0:
        raise ValueError("output_dt must be positive and finite")
    grouped = group_trajectory_segments(steps)
    if not any(isinstance(s, TrajectorySegment) for s in grouped):
        raise ValueError("Plan needs at least one motion leg")
    vel_cap = _limits(velocity_limits, len(joint_map.robot_joint_names), "velocity_limits")
    acc_cap = _limits(acceleration_limits, len(joint_map.robot_joint_names), "acceleration_limits")
    output = []
    for i, step in enumerate(grouped):
        if isinstance(step, GripperEvent):
            output.append(step)
            continue
        trajectory_segment_key = None
        if key is not None:
            key, trajectory_segment_key = jax.random.split(key)
        output.append(
            retime_trajectory_segment(
                encoder,
                step.positions,
                joint_map=joint_map,
                original_duration=step.duration,
                source_times=step.times,
                output_dt=output_dt,
                velocity_limits=vel_cap,
                acceleration_limits=acc_cap,
                lead_speed=0.0 if i == 0 else boundary_speed,
                trail_speed=0.0 if i == len(grouped) - 1 else boundary_speed,
                config=config,
                key=trajectory_segment_key,
                trajectory_validator=trajectory_validator,
            )
        )
    plan = _stitch_plan(output, vel_cap, acc_cap, output_dt=output_dt)
    if trajectory_validator is not None and not trajectory_validator(plan.positions, plan.times):
        raise RetimingError("Stitched trajectory failed final validation")
    return plan
