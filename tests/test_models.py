"""Alternate training embodiments use separate artifacts, dimensions and mappings."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import torch
from reference.torch_encoder import TrajectoryEncoder

from datafarm_pyroki import JointMap, load_encoder


def test_alternate_model_dimensions_and_order(tmp_path):
    torch.manual_seed(4)
    model = TrajectoryEncoder(ch=48, d=8, n_target=113).eval()
    artifact = tmp_path / "different_robot.npz"
    arrays = {
        f"state_dict/{key}": value.detach().numpy() for key, value in model.state_dict().items()
    }
    names = tuple(f"axis{i}" for i in range(12))
    arrays.update(
        schema_version=np.int32(1),
        model_name=np.asarray("test-robot-12"),
        joint_names=np.asarray(names),
        sample_rate_hz=np.asarray(20.0),
        chan_mu=np.zeros(48, np.float32),
        chan_sd=np.ones(48, np.float32),
        reference_mean=np.zeros(8, np.float32),
        reference_precision=np.eye(8, dtype=np.float32),
        reference_cov=np.eye(8, dtype=np.float32),
    )
    np.savez_compressed(artifact, **arrays)
    encoder = load_encoder(artifact)
    assert encoder.spec.n_joints == 12
    x = np.random.default_rng(5).normal(size=(1, 48, 30)).astype(np.float32)
    expected, _ = model.encode(torch.tensor(x), torch.ones(1, 1, 30))
    actual, _ = encoder.encode(jnp.asarray(x))
    np.testing.assert_allclose(actual, expected.detach().numpy(), atol=1e-6, rtol=1e-4)
    q = jnp.asarray(np.random.default_rng(6).normal(size=(10, 12)).astype(np.float32))
    assert encoder.preprocess(q, 0.1).shape == (1, 48, 19)
    score, grad = jax.jit(jax.value_and_grad(lambda q: encoder.cost(q, 0.1)))(q)
    assert np.isfinite(score)
    assert grad.shape == q.shape and np.isfinite(grad).all()
    mapping = JointMap.from_names(tuple(reversed(names)), names)
    np.testing.assert_array_equal(mapping.select(q[:, ::-1]), q)


def test_float64_waypoint_gradient_parity(encoder, torch_encoder, motion):
    """Higher precision isolates algorithm parity from jerk-amplified float32 roundoff."""
    from test_encoder import torch_preprocess

    with jax.enable_x64(True):
        precise = jax.tree.map(lambda x: x.astype(jnp.float64), encoder)
        model = copy.deepcopy(torch_encoder).double()
        q = torch.tensor(motion[None].astype(np.float64), requires_grad=True)
        x = torch_preprocess(precise, q, 0.15)
        mu, _ = model.encode(x, torch.ones(1, 1, x.shape[-1], dtype=torch.float64))
        dz = mu - torch.tensor(np.asarray(precise.latent_mean))
        cost = torch.einsum(
            "bi,ij,bj->", dz, torch.tensor(np.asarray(precise.latent_precision)), dz
        )
        cost.backward()
        actual, grad = jax.value_and_grad(lambda q: precise.cost(q, 0.15))(
            jnp.asarray(motion, dtype=jnp.float64)
        )
        # Casting the stored float32 Cholesky factor leaves a small fixed metric
        # roundoff; the coordinate-wise tolerance accounts for that factor only.
        np.testing.assert_allclose(actual, cost.detach().numpy(), rtol=2e-6)
        np.testing.assert_allclose(grad, q.grad.numpy()[0], rtol=2e-5, atol=0.005)
