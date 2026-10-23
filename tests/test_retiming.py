import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from datafarm_pyroki import JointMap, RetimeConfig, RetimingError, retime_stroke
from datafarm_pyroki.retiming import (
    clock_knots,
    motion_regularization,
    sample_clock,
    sample_path,
    timing_objective,
    timing_regularization,
)


@pytest.fixture
def stroke_input(encoder):
    t = np.linspace(0.0, 1.0, 16)[:, None]
    q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]) + 0.07 * np.sin(t * np.arange(1, 8)[None])
    mapping = JointMap.from_names(encoder.spec.joint_names, encoder.spec.joint_names)
    return q, mapping


@pytest.fixture
def short_config():
    return RetimeConfig(starts=2, iterations=8, knots=8)


def test_clock_and_path_preserve_sharp_corners():
    theta = jnp.array([[0.2, -0.3, 0.6]])
    tau = clock_knots(theta)
    assert np.all(np.diff(tau) > 0)
    np.testing.assert_allclose(tau[:, [0, -1]], [[0.0, 1.0]])
    s = sample_clock(tau, tau)
    np.testing.assert_allclose(s, jnp.linspace(0.0, 1.0, 4)[None], atol=1e-7)
    path = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    actual = sample_path(path, jnp.array([0.0, 0.5, 1.0]), jnp.array([0.25, 0.5, 0.75]))
    np.testing.assert_allclose(actual, [[0.5, 0.0], [1.0, 0.0], [1.0, 0.5]])


def test_quintic_clock_is_monotone_c2_and_stops_at_requested_ends():
    with jax.enable_x64(True):
        tau = clock_knots(jnp.array([[0.7, -0.4, 0.3, -0.8]], dtype=jnp.float64))

        def evaluate(t):
            return sample_clock(
                tau, t.reshape(1, 1), interpolation="quintic", rest_start=True, rest_end=True
            )[0, 0]

        velocity = jax.grad(evaluate)
        acceleration = jax.grad(velocity)
        grid = jnp.linspace(0, 1, 1001)
        values = jax.vmap(evaluate)(grid)
        assert np.all(np.diff(values) > 0)
        np.testing.assert_allclose(np.asarray(values)[[0, -1]], [0, 1], atol=1e-12)
        np.testing.assert_allclose(jax.vmap(velocity)(jnp.array([0.0, 1.0])), 0, atol=1e-12)
        # Both sides of every knot agree in value, speed and acceleration.
        eps = 1e-8
        for knot in tau[0, 1:-1]:
            for fn in (evaluate, velocity, acceleration):
                np.testing.assert_allclose(fn(knot - eps), fn(knot + eps), atol=2e-4)
        np.testing.assert_allclose(jax.vmap(acceleration)(jnp.array([0.0, 1.0])), 0, atol=1e-10)

        # Clock-parameter gradients must also propagate through the new sampler.
        def loss(theta):
            return sample_clock(
                clock_knots(theta), jnp.array([[0.27, 0.63]]), interpolation="quintic"
            ).sum()

        theta = jnp.array([[0.7, -0.4, 0.3, -0.8]])
        direction = jnp.array([[0.1, 0.3, -0.4, 0.2]])
        exact = jnp.sum(jax.grad(loss)(theta) * direction)
        finite = (loss(theta + 1e-5 * direction) - loss(theta - 1e-5 * direction)) / 2e-5
        np.testing.assert_allclose(exact, finite, rtol=1e-6)


def test_clock_regularization_penalizes_wiggles_and_excess_duration():
    config = RetimeConfig()
    uniform = clock_knots(jnp.zeros((1, 15)))
    wiggly = clock_knots(jnp.array([[0.5, -0.5] * 7 + [0.5]]))
    base = timing_regularization(uniform, 30.0, 30.0, config)
    assert float(timing_regularization(wiggly, 30.0, 30.0, config)[0]) > float(base[0]) + 5
    np.testing.assert_allclose(timing_regularization(uniform, 60.0, 30.0, config), 2 * base)


def test_motion_cost_uses_physical_derivatives_all_joints_and_ignores_padding():

    with jax.enable_x64(True):
        path = jnp.array([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
        arc = jnp.array([0.0, 1.0])
        clock = jnp.array([[0.0, 0.4, 1.0]])
        config = RetimeConfig(regularization_rate_hz=50.0)
        duration = jnp.array([1.3])

        def score(duration, n_frames=80, path=path):
            return motion_regularization(
                path,
                arc,
                clock,
                duration,
                n_frames=n_frames,
                lead_speed=0.0,
                trail_speed=0.0,
                config=config,
            )[0]

        # Independently finite-difference the physical position signal.
        frames = np.arange(80)
        q = np.asarray(
            sample_clock(
                clock,
                jnp.asarray(frames[None] / 65.0),
                interpolation="quintic",
                rest_start=True,
                rest_end=True,
            )
        )[0]
        expected = 0.0
        for order, weight in [(2, config.acceleration_weight), (3, config.jerk_weight)]:
            derivative = np.diff(q, n=order) * 50**order
            valid = np.clip(65.0 - frames[order:], 0, 1)
            expected += weight * np.sum(derivative**2 * valid) / (2 * valid.sum())
        np.testing.assert_allclose(score(duration), expected, rtol=1e-10)
        np.testing.assert_allclose(score(duration, 120), expected, rtol=1e-10)
        assert float(score(duration, path=path * 2)) > 3.9 * float(score(duration))
        # Interior-duration gradients include the physical timescale, rather than
        # only resampling a fixed number of frames at a different nominal dt.
        duration = jnp.array([1.307])
        exact = jax.grad(score)(duration)[0]
        finite = (score(duration + 1e-6) - score(duration - 1e-6)) / 2e-6
        np.testing.assert_allclose(exact, finite, rtol=1e-5)


def test_timing_cost_and_duration_profile_gradients_match_torch(encoder, torch_encoder):
    from test_encoder import torch_masked_encode

    with jax.enable_x64(True):
        precise = jax.tree.map(lambda x: x.astype(jnp.float64), encoder)
        model = copy.deepcopy(torch_encoder).double()
        rng = np.random.default_rng(8)
        path = rng.normal(size=(9, 7)) * 0.04
        arc = np.linspace(0.0, 1.0, len(path))
        theta = rng.normal(size=(2, 7)) * 0.1
        span = np.array([24.3, 31.7])
        config = RetimeConfig(knots=8, starts=2, iterations=2, clock_interpolation="linear")
        tp = torch.tensor(path)
        ta = torch.tensor(arc)
        tt = torch.tensor(theta, requires_grad=True)
        ts = torch.tensor(span, requires_grad=True)
        tau = torch.cat((torch.zeros(2, 1), torch.softmax(torch.tanh(tt), -1).cumsum(-1)), -1)
        tau = torch.cat((tau[:, :-1], torch.ones(2, 1)), -1)

        def torch_sample(fractions):
            index = torch.searchsorted(tau.contiguous(), fractions.contiguous(), right=True) - 1
            index = index.clamp(0, tau.shape[-1] - 2)
            left = tau.gather(1, index)
            right = tau.gather(1, index + 1)
            u = ((fractions - left) / (right - left)).clamp(0, 1)
            s = (index + u) / (tau.shape[-1] - 1)
            idx = (torch.searchsorted(ta, s.contiguous(), right=True) - 1).clamp(0, len(ta) - 2)
            v = ((s - ta[idx]) / (ta[idx + 1] - ta[idx])).clamp(0, 1)
            return tp[idx] + v[..., None] * (tp[idx + 1] - tp[idx])

        frames = torch.arange(48, dtype=torch.float64)
        q = torch_sample(frames[None] / ts[:, None])
        mask = (ts[:, None, None] - frames[None, None]).clamp(0, 1)

        def diff(x):
            return torch.cat(
                (
                    (x[:, 1:2] - x[:, :1]) * 15,
                    (x[:, 2:] - x[:, :-2]) * 7.5,
                    (x[:, -1:] - x[:, -2:-1]) * 15,
                ),
                dim=1,
            )

        v = diff(q)
        a = diff(v)
        features = torch.cat((q, v, a, diff(a)), -1)
        features = (
            (features - torch.tensor(np.asarray(precise.chan_mu)))
            / torch.tensor(np.asarray(precise.chan_sd))
        ).transpose(1, 2) * mask
        z, _ = torch_masked_encode(model, features, mask)
        dz = z - torch.tensor(np.asarray(precise.latent_mean))
        style = ((dz @ torch.tensor(np.asarray(precise.precision_cholesky))) ** 2).sum(-1)
        edge = torch_sample(
            torch.stack((torch.zeros_like(ts), torch.ones_like(ts), ts - 1, ts), -1) / ts[:, None]
        )
        boundary = ((edge[:, 1] - edge[:, 0]).norm(dim=-1) * 15 - 0.07) ** 2
        boundary += ((edge[:, 3] - edge[:, 2]).norm(dim=-1) * 15 - 0.08) ** 2
        loss = (style + 50 * boundary).sum()
        loss.backward()

        def objective(theta, span):
            style, bnd = timing_objective(
                precise,
                jnp.asarray(path),
                jnp.asarray(arc),
                theta,
                span,
                precise.latent_mean,
                joint_indices=tuple(range(7)),
                n_frames=48,
                lead_speed=0.07,
                trail_speed=0.08,
                config=config,
            )
            return (style + 50 * bnd).sum()

        value, (grad_theta, grad_span) = jax.value_and_grad(objective, argnums=(0, 1))(
            jnp.asarray(theta), jnp.asarray(span)
        )
        np.testing.assert_allclose(value, loss.detach().numpy(), rtol=1e-8)
        np.testing.assert_allclose(grad_theta, tt.grad.numpy(), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(grad_span, ts.grad.numpy(), rtol=1e-6, atol=1e-7)
        assert np.linalg.norm(grad_theta) > 0 and np.linalg.norm(grad_span) > 0


def test_actual_retimer_emits_feasible_motion(encoder, stroke_input, short_config):
    q, mapping = stroke_input
    result = retime_stroke(
        encoder,
        q,
        joint_map=mapping,
        original_duration=2.0,
        output_dt=0.02,
        velocity_limits=np.full(7, 3.0),
        acceleration_limits=np.full(7, 30.0),
        config=short_config,
        duration_bounds=(1.0, 3.0),
    )
    np.testing.assert_array_equal(result.positions[[0, -1]], q[[0, -1]])
    np.testing.assert_array_equal(result.path_positions, q)
    np.testing.assert_allclose(result.velocities[[0, -1]], 0)
    np.testing.assert_allclose(
        result.accelerations, np.gradient(result.velocities, result.dt, axis=0)
    )
    assert 1.0 <= result.duration <= 3.0 and np.all(np.diff(result.times) > 0)
    assert np.max(abs(result.velocities)) <= 3 and np.max(abs(result.accelerations)) <= 30
    assert np.isfinite(result.objective)
    np.testing.assert_allclose(
        result.objective,
        result.style_cost
        + short_config.boundary_weight * result.boundary_cost
        + result.regularization_cost,
    )
    # Clock shape and scale are separate optimized quantities.
    assert result.clock.shape == (8,) and np.all(np.diff(result.clock) > 0)
    assert np.isclose(result.times[-1], result.duration)


def test_retimer_rejects_caps_and_failed_validation(encoder, stroke_input, short_config):
    q, mapping = stroke_input
    kwargs = dict(
        joint_map=mapping,
        original_duration=2.0,
        output_dt=0.02,
        config=short_config,
        duration_bounds=(1.0, 3.0),
    )
    with pytest.raises(RetimingError, match="No emitted stroke"):
        retime_stroke(
            encoder, q, velocity_limits=np.full(7, 1e-8), acceleration_limits=np.ones(7), **kwargs
        )
    with pytest.raises(RetimingError, match="No emitted stroke"):
        retime_stroke(
            encoder,
            q,
            velocity_limits=np.full(7, 3.0),
            acceleration_limits=np.full(7, 30.0),
            trajectory_validator=lambda q, times: False,
            **kwargs,
        )
    with pytest.raises(ValueError, match="positive finite"):
        retime_stroke(
            encoder, q, velocity_limits=np.zeros(7), acceleration_limits=np.full(7, 30.0), **kwargs
        )


def test_explicit_target_and_random_key_are_reproducible(encoder, stroke_input):
    q, mapping = stroke_input
    kwargs = dict(
        joint_map=mapping,
        original_duration=2.0,
        output_dt=0.04,
        velocity_limits=np.full(7, 10.0),
        acceleration_limits=np.full(7, 1000.0),
        config=RetimeConfig(starts=1, iterations=2, knots=8),
        duration_bounds=(2.0, 2.0),
    )
    key = jax.random.key(3)
    a = retime_stroke(encoder, q, key=key, **kwargs)
    b = retime_stroke(encoder, q, key=key, **kwargs)
    np.testing.assert_array_equal(a.target_latent, b.target_latent)
    np.testing.assert_array_equal(a.positions, b.positions)
    expected = encoder.latent_mean + jnp.linalg.solve(
        encoder.precision_cholesky.T, jax.random.normal(key, encoder.latent_mean.shape)
    )
    np.testing.assert_allclose(a.target_latent, expected)
    with pytest.raises(ValueError, match="not both"):
        retime_stroke(encoder, q, key=key, target_latent=encoder.latent_mean, **kwargs)


def test_retimer_supports_other_joint_counts_and_checks_every_robot_joint(encoder):
    from dataclasses import replace

    from datafarm_pyroki import EncoderSpec

    # Synthetic expanded weights exercise the interface only; no transfer-learning
    # claim is made. Existing seven-coordinate filters occupy the corresponding
    # channels of a twelve-coordinate model; extra channels have zero weights.
    channel_indices = np.concatenate([np.arange(7) + group * 12 for group in range(4)])
    state = dict(encoder.state_dict)
    for name in [f"fb.{i}.weight" for i in range(6)] + ["t.0.weight"]:
        old = state[name]
        state[name] = jnp.zeros((old.shape[0], 48, old.shape[2])).at[:, channel_indices].set(old)
    mu = jnp.zeros(48).at[channel_indices].set(encoder.chan_mu)
    sd = jnp.ones(48).at[channel_indices].set(encoder.chan_sd)
    names = tuple(f"axis{i}" for i in range(12))
    custom = replace(
        encoder,
        state_dict=state,
        chan_mu=mu,
        chan_sd=sd,
        spec=EncoderSpec("shape-test-12", names, 15.0),
    )
    mapping = JointMap.from_names(tuple(reversed(names)), names)
    t = np.linspace(0.0, 1.0, 16)[:, None]
    q = 0.02 * np.sin(t * np.arange(1, 13)[None])
    robot_q = q[:, ::-1].copy()
    kwargs = dict(
        joint_map=mapping,
        original_duration=2.0,
        output_dt=0.04,
        acceleration_limits=np.full(12, 1000.0),
        config=RetimeConfig(starts=1, iterations=2, knots=8),
        duration_bounds=(1.0, 3.0),
    )
    result = retime_stroke(custom, robot_q, velocity_limits=np.full(12, 10.0), **kwargs)
    assert result.positions.shape[1] == 12
    np.testing.assert_array_equal(result.positions[[0, -1]], robot_q[[0, -1]])
    caps = np.full(12, 10.0)
    caps[0] = 1e-10  # coordinate outside the original seven still must obey its cap
    with pytest.raises(RetimingError):
        retime_stroke(custom, robot_q, velocity_limits=caps, **kwargs)
