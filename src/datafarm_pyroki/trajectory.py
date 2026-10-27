"""Robot-independent interpolation and validation of sampled joint trajectories."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class SampledTrajectory:
    positions: np.ndarray
    times: np.ndarray

    @property
    def dt(self) -> float:
        return float(self.times[1] - self.times[0])


class TrajectoryValidationError(ValueError):
    """Invalid sample data or a position, velocity, or acceleration limit violation."""


def _samples(positions, times):
    q, t = np.asarray(positions, dtype=np.float64), np.asarray(times, dtype=np.float64)
    if (
        q.ndim != 2
        or len(q) < 2
        or q.shape[1] == 0
        or t.shape != (len(q),)
        or not np.isfinite(q).all()
        or not np.isfinite(t).all()
        or np.any(np.diff(t) <= 0)
    ):
        raise TrajectoryValidationError(
            "Need finite positions [time>=2,joints] and increasing times"
        )
    return q, t


def interpolate_trajectory(
    positions, times, *, output_dt=0.02, ease_boundaries=False, ramp_fraction=0.1
) -> SampledTrajectory:
    """Sample a C2 joint spline with zero endpoint velocity and fixed duration.

    Optional boundary easing uses a uniform cruise with quintic speed ramps.
    Interpolation can change geometry between source samples; callers must check
    task/collision feasibility on the resulting path.
    """
    q, t = _samples(positions, times)
    if not np.isfinite(output_dt) or output_dt <= 0:
        raise ValueError("output_dt must be positive and finite")
    if not np.isfinite(ramp_fraction) or not 0 < ramp_fraction <= 0.5:
        raise ValueError("ramp_fraction must be in (0,0.5]")
    t = t - t[0]
    zeros = np.zeros(q.shape[1])
    count = max(3, int(round(t[-1] / output_dt)) + 1)
    clock = np.linspace(0, t[-1], count)
    spline = CubicSpline(t, q, bc_type=((1, zeros), (1, zeros)))
    positions = spline(clock)
    if ease_boundaries:
        # Preserve the existing two-stage interpolation: source curve, then clock.
        u, ramp = clock / clock[-1], ramp_fraction

        def integral(x):
            return x**6 - 3 * x**5 + 2.5 * x**4

        progress = np.where(
            u < ramp,
            ramp * integral(np.clip(u / ramp, 0, 1)) / (1 - ramp),
            np.where(
                u > 1 - ramp,
                1 - ramp * integral(np.clip((1 - u) / ramp, 0, 1)) / (1 - ramp),
                (u - ramp / 2) / (1 - ramp),
            ),
        )
        positions = CubicSpline(clock, positions, bc_type=((1, zeros), (1, zeros)))(
            clock[-1] * progress
        )
    positions[0], positions[-1] = q[0], q[-1]
    return SampledTrajectory(positions, clock)


def validate_trajectory(
    positions,
    times,
    *,
    velocity_limits,
    acceleration_limits,
    lower_limits=None,
    upper_limits=None,
    position_tolerance=0.0,
    rest_endpoints=True,
):
    """Check joint samples and interval speeds, returning velocity/acceleration arrays.

    Limits are explicit arrays in the trajectory's coordinate order; no robot or
    joint-count assumptions are made. Timestamp spacing may be nonuniform.
    Rest endpoints zero the endpoint velocity estimates before differentiating.
    """
    q, t = _samples(positions, times)
    if not np.isfinite(position_tolerance) or position_tolerance < 0:
        raise ValueError("position_tolerance must be nonnegative and finite")
    dof = q.shape[1]

    def caps(value, name):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (dof,) or np.any(array <= 0) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain one positive finite limit per joint")
        return array

    velocity_limits = caps(velocity_limits, "velocity_limits")
    acceleration_limits = caps(acceleration_limits, "acceleration_limits")
    if (lower_limits is None) != (upper_limits is None):
        raise ValueError("Supply both lower_limits and upper_limits")
    if lower_limits is not None:
        lower, upper = np.asarray(lower_limits), np.asarray(upper_limits)
        if (
            lower.shape != (dof,)
            or upper.shape != (dof,)
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(lower > upper)
        ):
            raise ValueError("Position limits must be finite ordered arrays in joint order")
        if np.any(q < lower - position_tolerance) or np.any(q > upper + position_tolerance):
            raise TrajectoryValidationError("Trajectory violates joint position limits")
    velocity = np.gradient(q, t, axis=0)
    if rest_endpoints:
        velocity[[0, -1]] = 0.0
    acceleration = np.gradient(velocity, t, axis=0)
    interval_speed = np.diff(q, axis=0) / np.diff(t)[:, None]
    if (
        not np.isfinite(velocity).all()
        or not np.isfinite(acceleration).all()
        or np.any(np.abs(interval_speed) > velocity_limits)
        or np.any(np.abs(velocity) > velocity_limits)
        or np.any(np.abs(acceleration) > acceleration_limits)
    ):
        raise TrajectoryValidationError("Trajectory violates velocity or acceleration limits")
    return velocity, acceleration
