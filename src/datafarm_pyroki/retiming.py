"""Standalone JAX trajectory segment timing, following TiPToP's encoder-based retiming design.

TiPToP algorithm attribution: Copyright (c) 2025 TiPToP Contributors (MIT).
See licenses/TIPTOP.txt. This implementation has no TiPToP/cuRobo/PyTorch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .embodiment import JointMap
from .encoder import MotionEncoder


@dataclass(frozen=True)
class RetimeConfig:
    """Static optimizer settings. Durations are seconds; speeds use model joint units/s."""

    knots: int = 64
    starts: int = 16
    iterations: int = 500
    profile_rail: float = 1.0
    profile_learning_rate: float = 0.02
    duration_learning_rate: float = 0.005  # fraction of allowed span range per Adam step
    boundary_weight: float = 50.0
    clock_interpolation: str = "quintic"
    profile_smoothness_weight: float = 0.1
    profile_deviation_weight: float = 0.5
    duration_weight: float = 2.0  # duration / original duration
    acceleration_weight: float = 0.1  # mean squared robot-coordinate acceleration
    jerk_weight: float = 0.01  # mean squared robot-coordinate jerk
    regularization_rate_hz: float = 50.0
    min_duration: float = 0.8
    max_duration: float = 20.0
    min_duration_multiplier: float = 0.2
    max_duration_multiplier: float = 2.0
    smoothing: float = 0.0  # zero preserves the original polyline as the continuous path

    def __post_init__(self):
        if any(
            not isinstance(x, int) or isinstance(x, bool) or x < lo
            for x, lo in ((self.knots, 3), (self.starts, 1), (self.iterations, 1))
        ):
            raise ValueError("knots >= 3, starts >= 1 and iterations >= 1 must be integers")
        positive = (
            self.profile_rail,
            self.profile_learning_rate,
            self.duration_learning_rate,
            self.min_duration,
            self.max_duration,
            self.min_duration_multiplier,
            self.max_duration_multiplier,
            self.regularization_rate_hz,
        )
        if not all(np.isfinite(x) and x > 0 for x in positive):
            raise ValueError(
                "Retiming scales, rates and duration bounds must be positive and finite"
            )
        if self.max_duration < self.min_duration:
            raise ValueError("max_duration must be at least min_duration")
        if self.min_duration_multiplier > self.max_duration_multiplier:
            raise ValueError("Duration multipliers must be ordered")
        if self.clock_interpolation not in ("linear", "quintic"):
            raise ValueError("clock_interpolation must be linear or quintic")
        weights = (
            self.boundary_weight,
            self.smoothing,
            self.profile_smoothness_weight,
            self.profile_deviation_weight,
            self.duration_weight,
            self.acceleration_weight,
            self.jerk_weight,
        )
        if not all(np.isfinite(x) and x >= 0 for x in weights):
            raise ValueError("Retiming weights and smoothing must be nonnegative and finite")


class RetimingError(RuntimeError):
    """No timing candidate meets the emitted-motion limits/validation requirements."""


@dataclass(frozen=True)
class RetimedTrajectorySegment:
    positions: np.ndarray  # [T, robot_dof], in ROBOT order
    velocities: np.ndarray
    accelerations: np.ndarray
    times: np.ndarray  # local seconds from zero
    duration: float
    dt: float  # actual uniform timestep, close to requested output_dt
    style_cost: float  # unweighted emitted motion score against the selected target
    boundary_cost: float  # unweighted squared speed-target errors
    objective: float
    original_style_cost: float
    target_latent: np.ndarray
    path_positions: np.ndarray  # fixed continuous geometric polyline used by the optimizer
    path_progress: np.ndarray
    clock: np.ndarray  # normalized arrival times at uniformly spaced arc-fraction knots
    lead_speed: float
    trail_speed: float
    regularization_cost: float = 0.0
    motion_cost: float = 0.0


def clock_knots(theta: jax.Array, rail: float = 1.0) -> jax.Array:
    """Positive interval time shares -> monotone normalized arrival times [B,K]."""
    shares = jax.nn.softmax(rail * jnp.tanh(theta), axis=-1)
    tau = jnp.pad(jnp.cumsum(shares, axis=-1), ((0, 0), (1, 0)))
    return tau.at[:, -1].set(1.0)


def sample_clock(
    tau: jax.Array,
    time_fraction: jax.Array,
    *,
    interpolation: str = "linear",
    rest_start: bool = False,
    rest_end: bool = False,
) -> jax.Array:
    """Monotone time -> progress, optionally C2 with zero speed at requested ends.

    Quintic Bezier control points use shared knot slopes and zero second
    derivatives. Slopes bounded by adjacent secants keep the control polygon
    monotone, so smoothing cannot reverse motion or overshoot the path endpoints.
    """
    k = tau.shape[-1]
    index = jnp.clip((time_fraction[..., None] >= tau[:, None, :-1]).sum(-1) - 1, 0, k - 2)
    left = jnp.take_along_axis(tau, index, axis=1)
    right = jnp.take_along_axis(tau, index + 1, axis=1)
    fraction = jnp.clip((time_fraction - left) / jnp.maximum(right - left, 1e-9), 0, 1)
    if interpolation == "linear":
        return (index + fraction) / (k - 1)
    secants = 1 / ((k - 1) * jnp.maximum(jnp.diff(tau, axis=-1), 1e-9))
    slopes = jnp.concatenate(
        (secants[:, :1], jnp.minimum(secants[:, :-1], secants[:, 1:]), secants[:, -1:]), -1
    )
    if rest_start:
        slopes = slopes.at[:, 0].set(0)
    if rest_end:
        slopes = slopes.at[:, -1].set(0)
    d0 = jnp.take_along_axis(slopes, index, axis=1) * (right - left)
    d1 = jnp.take_along_axis(slopes, index + 1, axis=1) * (right - left)
    y0, y1 = index / (k - 1), (index + 1) / (k - 1)
    control = jnp.stack((y0, y0 + d0 / 5, y0 + 2 * d0 / 5, y1 - 2 * d1 / 5, y1 - d1 / 5, y1), -1)
    # de Casteljau evaluation is stable at knots and differentiable in the clock.
    for _ in range(5):
        control = (1 - fraction[..., None]) * control[..., :-1] + fraction[..., None] * control[
            ..., 1:
        ]
    return control[..., 0]


def timing_regularization(clock, span, reference_span, config):
    """Dimensionless profile variation and elapsed-time costs, per candidate."""
    log_shares = jnp.log(jnp.maximum(jnp.diff(clock, axis=-1), 1e-9))
    centered = log_shares - jnp.mean(log_shares, axis=-1, keepdims=True)
    # Scale the finite difference by profile resolution so changing resolution does not
    # silently weaken the prior against rapid changes in local path speed.
    roughness = jnp.mean((jnp.diff(log_shares, axis=-1) * (log_shares.shape[-1] - 1)) ** 2, -1)
    deviation = jnp.mean(centered**2, -1)
    return (
        config.profile_smoothness_weight * roughness
        + config.profile_deviation_weight * deviation
        + config.duration_weight * span / reference_span
    )


def motion_regularization(
    path, arc, clock, durations, *, n_frames, lead_speed, trail_speed, config
):
    """Physical acceleration/jerk costs in all robot coordinates, per candidate.

    This fixed physical grid is independent of the trained encoder rate. A
    prefix mask excludes difference stencils extending beyond the trajectory segment.
    """
    rate = config.regularization_rate_hz
    frames = jnp.arange(n_frames, dtype=path.dtype)
    progress = sample_clock(
        clock,
        frames[None] / (durations[:, None] * rate),
        interpolation=config.clock_interpolation,
        rest_start=lead_speed == 0,
        rest_end=trail_speed == 0,
    )
    q = sample_path(path, arc, progress)
    cost = jnp.zeros_like(durations)
    for order, weight in ((2, config.acceleration_weight), (3, config.jerk_weight)):
        derivative = jnp.diff(q, n=order, axis=1) * rate**order
        valid = jnp.clip(durations[:, None] * rate - frames[None, order:], 0, 1)
        mean_square = (derivative**2 * valid[..., None]).sum((1, 2)) / jnp.maximum(
            valid.sum(-1) * q.shape[-1], 1
        )
        cost = cost + weight * mean_square
    return cost


def sample_path(path: jax.Array, arc: jax.Array, progress: jax.Array) -> jax.Array:
    """Evaluate a fixed polyline without replacing its corners by clock-knot chords."""
    index = jnp.clip(jnp.searchsorted(arc, progress, side="right") - 1, 0, len(arc) - 2)
    fraction = jnp.clip(
        (progress - arc[index]) / jnp.maximum(arc[index + 1] - arc[index], 1e-12), 0, 1
    )
    return path[index] + fraction[..., None] * (path[index + 1] - path[index])


def timing_objective(
    encoder: MotionEncoder,
    path: jax.Array,
    arc: jax.Array,
    theta: jax.Array,
    span: jax.Array,
    target: jax.Array,
    *,
    joint_indices: tuple[int, ...],
    n_frames: int,
    lead_speed: float,
    trail_speed: float,
    config: RetimeConfig,
) -> tuple[jax.Array, jax.Array]:
    """Per-start (style, boundary) costs on a FIXED model-rate grid.

    span is duration in model-rate frame intervals, not a sample count rounded to
    an integer. A soft prefix mask makes its effect differentiable. The encoder's
    temporal filters keep their trained physical timescale throughout optimization.
    """
    frames = jnp.arange(n_frames, dtype=path.dtype)
    tau = clock_knots(theta, config.profile_rail)
    sampling = dict(
        interpolation=config.clock_interpolation,
        rest_start=lead_speed == 0,
        rest_end=trail_speed == 0,
    )
    progress = sample_clock(tau, frames[None] / span[:, None], **sampling)
    q = sample_path(path, arc, progress)[..., jnp.asarray(joint_indices)]
    mask = jnp.clip(span[:, None, None] - frames[None, None], 0.0, 1.0)
    features = encoder.features_at_model_rate(q) * mask
    latent, _ = encoder.encode(features, mask, padding_exact=True)
    dz = latent - target
    style = jnp.sum((dz @ encoder.precision_cholesky) ** 2, axis=-1)
    boundary = jnp.zeros_like(style)
    endpoint_frames = jnp.stack((jnp.zeros_like(span), jnp.ones_like(span), span - 1, span), -1)
    endpoint_progress = sample_clock(tau, endpoint_frames / span[:, None], **sampling)
    endpoint_q = sample_path(path, arc, endpoint_progress)[..., jnp.asarray(joint_indices)]
    for a, b, speed in ((0, 1, lead_speed), (2, 3, trail_speed)):
        if speed > 0:
            # A clamped norm has a finite gradient even for a stationary path.
            norm = jnp.sqrt(
                jnp.maximum(jnp.sum((endpoint_q[:, b] - endpoint_q[:, a]) ** 2, -1), 1e-16)
            )
            boundary = boundary + (norm * encoder.spec.sample_rate_hz - speed) ** 2
    return style, boundary


class _Candidates(NamedTuple):
    theta: jax.Array
    span: jax.Array
    finite: jax.Array


@partial(
    jax.jit, static_argnames=("joint_indices", "n_frames", "lead_speed", "trail_speed", "config")
)
def _optimize(
    encoder,
    path,
    arc,
    spans,
    lower_span,
    upper_span,
    reference_span,
    target,
    *,
    joint_indices,
    n_frames,
    lead_speed,
    trail_speed,
    config,
):
    def objective(params):
        style, boundary = timing_objective(
            encoder,
            path,
            arc,
            params["theta"],
            params["span"],
            target,
            joint_indices=joint_indices,
            n_frames=n_frames,
            lead_speed=lead_speed,
            trail_speed=trail_speed,
            config=config,
        )
        regularization = timing_regularization(
            clock_knots(params["theta"], config.profile_rail),
            params["span"],
            reference_span,
            config,
        )
        regularization = regularization + motion_regularization(
            path,
            arc,
            clock_knots(params["theta"], config.profile_rail),
            params["span"] / encoder.spec.sample_rate_hz,
            n_frames=int(
                np.ceil(
                    (n_frames - 8) / encoder.spec.sample_rate_hz * config.regularization_rate_hz
                )
            )
            + 8,
            lead_speed=lead_speed,
            trail_speed=trail_speed,
            config=config,
        )
        total = style + config.boundary_weight * boundary + regularization
        return total.sum(), total

    initial = {"theta": jnp.zeros((len(spans), config.knots - 1), dtype=path.dtype), "span": spans}
    zero = jax.tree.map(jnp.zeros_like, initial)
    _, initial_values = objective(initial)
    initial_valid = jnp.isfinite(initial_values)
    rates = {
        "theta": config.profile_learning_rate,
        "span": config.duration_learning_rate * (upper_span - lower_span),
    }

    def step(carry, i):
        params, m, v, best, best_values, valid = carry
        (_, values), grad = jax.value_and_grad(objective, has_aux=True)(params)
        valid = valid & jnp.isfinite(values)
        valid = valid & jnp.isfinite(grad["theta"]).all(-1) & jnp.isfinite(grad["span"])
        grad = {
            "theta": jnp.where(valid[:, None], grad["theta"], 0),
            "span": jnp.where(valid, grad["span"], 0),
        }
        improve = valid & (values < best_values)
        best = {
            "theta": jnp.where(improve[:, None], params["theta"], best["theta"]),
            "span": jnp.where(improve, params["span"], best["span"]),
        }
        best_values = jnp.minimum(jnp.where(valid, values, jnp.inf), best_values)
        m = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, m, grad)
        v = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g * g, v, grad)
        count = i + 1
        params = jax.tree.map(
            lambda p, m, v, lr: (
                p - lr * (m / (1 - 0.9**count)) / (jnp.sqrt(v / (1 - 0.999**count)) + 1e-8)
            ),
            params,
            m,
            v,
            rates,
        )
        params["span"] = jnp.clip(params["span"], lower_span, upper_span)
        return (params, m, v, best, best_values, valid), None

    (final, _, _, best, _, valid), _ = jax.lax.scan(
        step,
        (initial, zero, zero, initial, initial_values, initial_valid),
        jnp.arange(config.iterations),
    )
    # Retain initial starts: an otherwise useful motion is not lost solely because
    # optimization moved a feasible clock outside the post-emission limits.
    return _Candidates(
        jnp.concatenate((best["theta"], final["theta"], initial["theta"])),
        jnp.concatenate((best["span"], final["span"], initial["span"])),
        jnp.concatenate((valid, valid, initial_valid)),
    )


def _prepare_geometry(q: np.ndarray, config: RetimeConfig) -> tuple[np.ndarray, np.ndarray]:
    # Remove only coincident consecutive points; preserve genuine path corners.
    keep = np.r_[True, np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-10]
    positions = q[keep]
    if len(positions) == 1:
        return np.repeat(positions, 2, axis=0), np.array([0.0, 1.0])
    if not np.array_equal(positions[-1], q[-1]):
        positions[-1] = q[-1]
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    if config.smoothing > 0:
        from scipy.interpolate import CubicSpline, make_smoothing_spline

        grid = np.linspace(0.0, distance[-1], max(256, len(positions) * 4, config.knots))
        columns = []
        for j in range(q.shape[1]):
            if len(positions) >= 5:
                spline = make_smoothing_spline(distance, positions[:, j], lam=config.smoothing)
            else:
                spline = CubicSpline(distance, positions[:, j], bc_type="natural")
            columns.append(spline(grid))
        positions = np.stack(columns, -1)
        positions[0], positions[-1] = q[0], q[-1]
        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
        keep = np.r_[True, np.diff(distance) > 1e-12]
        positions, distance = positions[keep], distance[keep]
    return positions, distance / distance[-1]


def _limits(limits, dof, name):
    values = np.asarray(limits, dtype=np.float64)
    if values.shape != (dof,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{name} must contain one positive finite value per robot joint")
    return values


def _score_emitted(encoder, positions, times, mapping, target):
    count = max(2, int(round(times[-1] * encoder.spec.sample_rate_hz)) + 1)
    grid = np.linspace(0.0, times[-1], count)
    q = np.stack([np.interp(grid, times, positions[:, j]) for j in mapping.indices], -1)
    features = encoder.features_at_model_rate(jnp.asarray(q[None], dtype=jnp.float32))
    z, _ = encoder.encode(features)
    residual = (z[0] - target) @ encoder.precision_cholesky
    return float(jnp.sum(residual**2))


def _boundary_cost(positions, times, mapping, lead_speed, trail_speed, rate):
    points = np.array([0.0, min(1 / rate, times[-1]), max(0.0, times[-1] - 1 / rate), times[-1]])
    q = np.stack([np.interp(points, times, positions[:, j]) for j in mapping.indices], -1)
    result = 0.0
    for a, b, speed in ((0, 1, lead_speed), (2, 3, trail_speed)):
        if speed > 0:
            result += (np.linalg.norm(q[b] - q[a]) * rate - speed) ** 2
    return float(result)


def retime_trajectory_segment(
    encoder: MotionEncoder,
    positions,
    *,
    joint_map: JointMap,
    original_duration: float,
    output_dt: float,
    velocity_limits,
    acceleration_limits,
    lead_speed: float = 0.0,
    trail_speed: float = 0.0,
    config: RetimeConfig = RetimeConfig(),
    duration_bounds: tuple[float, float] | None = None,
    source_times=None,
    target_latent=None,
    key: jax.Array | None = None,
    trajectory_validator: Callable[[np.ndarray, np.ndarray], bool] | None = None,
) -> RetimedTrajectorySegment:
    """Optimize only a trajectory segment's clock, then select a limit-feasible emitted candidate.

    Default target is the reference mean. `target_latent` supplies a fixed target;
    alternatively `key` samples once from the reference Gaussian, shared by all
    starts and final ranking (no hidden/global RNG). Output is in robot joint order.

    `source_times`, if supplied, are local timestamps of the input path's original
    execution; otherwise input points are assumed uniformly spaced over original_duration.
    Lead/trail speeds are soft targets in MODEL coordinates. With the default
    quintic clock, zero eases the position trajectory to rest; emitted endpoint
    velocity is also pinned. Acceleration is recomputed after pinning and checked.
    Smoothing changes geometry explicitly; validate collision/task feasibility with
    trajectory_validator. It receives the emitted positions and timestamps.
    """
    q = np.asarray(positions, dtype=np.float64)
    if q.ndim != 2 or len(q) < 2 or q.shape[-1] != len(joint_map.robot_joint_names):
        raise ValueError("Expected positions [time>=2,robot_joints] matching JointMap")
    if not np.isfinite(q).all():
        raise ValueError("Positions must be finite")
    if joint_map.model_joint_names != encoder.spec.joint_names:
        raise ValueError("JointMap does not match the encoder")
    if not all(np.isfinite(x) and x > 0 for x in (original_duration, output_dt)):
        raise ValueError("original_duration and output_dt must be positive and finite")
    if not all(np.isfinite(x) and x >= 0 for x in (lead_speed, trail_speed)):
        raise ValueError("Boundary speeds must be nonnegative and finite")
    vel_cap = _limits(velocity_limits, q.shape[1], "velocity_limits")
    acc_cap = _limits(acceleration_limits, q.shape[1], "acceleration_limits")
    source_times = (
        np.linspace(0.0, original_duration, len(q))
        if source_times is None
        else np.asarray(source_times, dtype=np.float64)
    )
    if (
        source_times.shape != (len(q),)
        or not np.isfinite(source_times).all()
        or source_times[0] != 0
        or not np.isclose(source_times[-1], original_duration)
        or np.any(np.diff(source_times) <= 0)
    ):
        raise ValueError("source_times must increase from zero to original_duration")
    if duration_bounds is None:
        lower = max(config.min_duration, config.min_duration_multiplier * original_duration)
        upper = min(config.max_duration, config.max_duration_multiplier * original_duration)
        # Very short source motions can legitimately be lengthened to the minimum.
        if upper < lower and lower <= config.max_duration:
            upper = lower
    else:
        lower, upper = duration_bounds
    if not np.isfinite([lower, upper]).all() or lower <= 0 or upper < lower:
        raise ValueError("Duration bounds must be positive, finite and ordered")
    if lower * encoder.spec.sample_rate_hz < 2:
        raise ValueError("Minimum duration must span at least two model-rate intervals")
    if target_latent is not None and key is not None:
        raise ValueError("Choose either target_latent or key, not both")
    target = encoder.latent_mean
    if key is not None:
        # P=C C.T -> covariance draw = C^{-T} epsilon.
        noise = jax.random.normal(key, encoder.latent_mean.shape)
        target = target + jnp.linalg.solve(encoder.precision_cholesky.T, noise)
    elif target_latent is not None:
        target = jnp.asarray(target_latent, dtype=jnp.float32)
    if target.shape != encoder.latent_mean.shape or not np.isfinite(target).all():
        raise ValueError("target_latent must be finite and match the latent dimension")

    path, arc = _prepare_geometry(q, config)
    rate = encoder.spec.sample_rate_hz
    starts = np.geomspace(lower, upper, config.starts + 2)[1:-1] * rate
    candidates = _optimize(
        encoder,
        jnp.asarray(path, dtype=jnp.float32),
        jnp.asarray(arc, dtype=jnp.float32),
        jnp.asarray(starts, dtype=jnp.float32),
        jnp.float32(lower * rate),
        jnp.float32(upper * rate),
        jnp.float32(original_duration * rate),
        target,
        joint_indices=joint_map.indices,
        n_frames=int(np.ceil(upper * rate)) + 8,
        lead_speed=lead_speed,
        trail_speed=trail_speed,
        config=config,
    )
    clocks = np.asarray(clock_knots(candidates.theta, config.profile_rail))
    durations = np.clip(np.asarray(candidates.span, dtype=np.float64) / rate, lower, upper)
    valid = np.asarray(candidates.finite)
    best = None
    for duration, clock, finite in zip(durations, clocks, valid, strict=True):
        if not finite or not np.isfinite(duration) or not np.isfinite(clock).all():
            continue
        count = max(3, int(round(duration / output_dt)) + 1)
        times = np.linspace(0.0, duration, count)
        progress = np.asarray(
            sample_clock(
                jnp.asarray(clock[None]),
                jnp.asarray((times / duration)[None]),
                interpolation=config.clock_interpolation,
                rest_start=lead_speed == 0,
                rest_end=trail_speed == 0,
            )
        )[0]
        emitted = np.stack([np.interp(progress, arc, path[:, j]) for j in range(q.shape[1])], -1)
        emitted[0], emitted[-1] = q[0], q[-1]
        dt = float(times[1])
        velocities = np.gradient(emitted, dt, axis=0)
        if lead_speed == 0:
            velocities[0] = 0.0
        if trail_speed == 0:
            velocities[-1] = 0.0
        accelerations = np.gradient(velocities, dt, axis=0)
        # Check both emitted derivatives and interval speeds; endpoint pinning may not hide a spike.
        interval_speeds = np.diff(emitted, axis=0) / dt
        if (
            not np.isfinite(velocities).all()
            or not np.isfinite(accelerations).all()
            or np.any(np.abs(interval_speeds) > vel_cap)
            or np.any(np.abs(velocities) > vel_cap)
            or np.any(np.abs(accelerations) > acc_cap)
        ):
            continue
        if trajectory_validator is not None and not trajectory_validator(emitted, times):
            continue
        style = _score_emitted(encoder, emitted, times, joint_map, target)
        boundary = _boundary_cost(emitted, times, joint_map, lead_speed, trail_speed, rate)
        regularization = float(
            timing_regularization(jnp.asarray(clock[None]), duration, original_duration, config)[0]
        )
        motion_cost = config.acceleration_weight * np.mean(
            (np.diff(emitted, n=2, axis=0) / dt**2) ** 2
        )
        if len(emitted) >= 4:
            motion_cost += config.jerk_weight * np.mean(
                (np.diff(emitted, n=3, axis=0) / dt**3) ** 2
            )
        regularization += float(motion_cost)
        objective = style + config.boundary_weight * boundary + regularization
        if np.isfinite(objective) and (best is None or objective < best[0]):
            best = (
                objective,
                emitted,
                times,
                velocities,
                accelerations,
                style,
                boundary,
                clock,
                regularization,
                float(motion_cost),
            )
    if best is None:
        raise RetimingError(
            "No emitted trajectory segment satisfies motion limits and trajectory validation "
            f"within duration bounds [{lower:.3g}, {upper:.3g}] seconds"
        )
    (
        objective,
        emitted,
        times,
        velocities,
        accelerations,
        style,
        boundary,
        clock,
        regularization,
        motion_cost,
    ) = best
    original_style = _score_emitted(encoder, q, source_times, joint_map, target)
    return RetimedTrajectorySegment(
        emitted,
        velocities,
        accelerations,
        times,
        float(times[-1]),
        float(times[1]),
        style,
        boundary,
        objective,
        original_style,
        np.asarray(target),
        path,
        arc,
        clock,
        lead_speed,
        trail_speed,
        regularization,
        motion_cost,
    )
