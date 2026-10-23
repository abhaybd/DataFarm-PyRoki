"""Parity with the MIT-licensed upstream PyTorch model, using the actual shipped weights."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from datafarm_pyroki.encoder import sample_count


def torch_preprocess(encoder, q, dt):
    count = sample_count(q.shape[1], dt)
    q = F.interpolate(q.transpose(1, 2), size=count, mode="linear", align_corners=True)
    q = q.transpose(1, 2)

    def diff(x):
        h = 1 / 15
        return torch.cat(
            (
                (x[:, 1:2] - x[:, :1]) / h,
                (x[:, 2:] - x[:, :-2]) / (2 * h),
                (x[:, -1:] - x[:, -2:-1]) / h,
            ),
            dim=1,
        )

    v = diff(q)
    a = diff(v)
    jerk = diff(a)
    raw = torch.cat((q, v, a, jerk), dim=-1)
    return (
        (raw - torch.tensor(np.array(encoder.chan_mu))) / torch.tensor(np.array(encoder.chan_sd))
    ).transpose(1, 2)


def torch_masked_encode(model, x, mask):
    """Independent evaluation of the runtime's prefix-masked temporal branch."""
    from reference.torch_encoder import masked_mean_std

    pooled = []
    for conv, norm in zip(model.fb, model.fbn):
        pooled.extend(masked_mean_std(norm(conv(x)).abs(), mask))
    features = x
    for layer in model.t:
        features = layer(features)
        if isinstance(layer, torch.nn.Conv1d):
            mask = mask[:, :, :: layer.stride[0]][:, :, : features.shape[-1]]
        features = features * mask
    mean, std = masked_mean_std(features, mask)
    maximum = (features + ((mask > 0.5).to(features.dtype) - 1) * 1e9).amax(-1)
    return model.to_lat(model.fc(torch.cat([*pooled, mean, std, maximum], dim=1))).chunk(2, 1)


@pytest.mark.parametrize("length", [2, 32, 70, 95])
def test_latent_and_auxiliary_head_match_pytorch(encoder, torch_encoder, length):
    x = np.random.default_rng(length).normal(size=(2, 28, length)).astype(np.float32)
    mask = np.ones((2, 1, length), np.float32)
    with torch.no_grad():
        mu, logvar = torch_encoder.encode(torch.tensor(x), torch.tensor(mask))
        aux = torch_encoder.aux(mu)
    actual_mu, actual_logvar = encoder.encode(jnp.asarray(x))
    np.testing.assert_allclose(actual_mu, mu.numpy(), atol=8e-6, rtol=2e-4)
    np.testing.assert_allclose(actual_logvar, logvar.numpy(), atol=8e-6, rtol=2e-4)
    np.testing.assert_allclose(
        encoder.predict_fingerprint(actual_mu), aux.numpy(), atol=1e-5, rtol=2e-4
    )


@pytest.mark.parametrize("padding_exact", [False, True])
def test_masked_outputs_and_feature_gradients(encoder, torch_encoder, padding_exact):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(2, 28, 80)).astype(np.float32)
    mask = np.ones((2, 1, 80), dtype=np.float32)
    mask[0, :, 53:] = 0
    mask[1, :, 71:] = 0
    mask[0, :, 52] = 0.4  # exercise a soft retiming mask
    x *= mask
    tx = torch.tensor(x, requires_grad=True)
    tm = torch.tensor(mask)
    if padding_exact:
        tmu, _ = torch_masked_encode(torch_encoder, tx, tm)
    else:
        tmu, _ = torch_encoder.encode(tx, tm)
    tmu.square().sum().backward()

    def objective(x):
        mu, _ = encoder.encode(x, jnp.asarray(mask), padding_exact=padding_exact)
        return jnp.sum(mu**2), mu

    (_, mu), grad = jax.value_and_grad(objective, has_aux=True)(jnp.asarray(x))
    np.testing.assert_allclose(mu, tmu.detach().numpy(), atol=1e-5, rtol=3e-4)
    np.testing.assert_allclose(grad, tx.grad.numpy(), atol=3e-5, rtol=8e-4)


@pytest.mark.parametrize("dt", [1 / 15, 0.15, 0.1])
def test_preprocessing_latent_cost_and_waypoint_gradients(encoder, torch_encoder, motion, dt):
    q = torch.tensor(motion[None], requires_grad=True)
    x = torch_preprocess(encoder, q, dt)
    mu, _ = torch_encoder.encode(x, torch.ones(1, 1, x.shape[-1]))
    delta = mu - torch.tensor(np.array(encoder.latent_mean))
    precision = torch.tensor(np.array(encoder.latent_precision))
    cost = torch.einsum("bi,ij,bj->", delta, precision, delta)
    cost.backward()
    actual_x = encoder.preprocess(jnp.asarray(motion), dt)
    np.testing.assert_allclose(actual_x, x.detach().numpy(), atol=3e-4, rtol=5e-4)
    actual_cost, grad = jax.jit(jax.value_and_grad(lambda p: encoder.cost(p, dt)))(
        jnp.asarray(motion)
    )
    np.testing.assert_allclose(actual_cost, cost.detach().numpy(), atol=0.03, rtol=1e-4)
    # Third differences amplify float32 interpolation roundoff; near-zero band
    # activations can also switch abs/max branches. Compare the full gradient field.
    expected_grad = q.grad.numpy()[0]
    relative_l2 = np.linalg.norm(np.asarray(grad) - expected_grad) / np.linalg.norm(expected_grad)
    assert relative_l2 < 1e-3, relative_l2
    cosine = np.vdot(grad, expected_grad) / (np.linalg.norm(grad) * np.linalg.norm(expected_grad))
    assert cosine > 0.999999
    assert np.isfinite(grad).all()


def test_whitening_and_weight_scaling(encoder, motion):
    q = jnp.stack([motion, motion + 0.02])
    mu = encoder.latent(q, 0.15)
    delta = mu - encoder.latent_mean
    direct = jnp.einsum("bi,ij,bj->b", delta, encoder.latent_precision, delta)
    np.testing.assert_allclose(encoder.cost(q, 0.15), direct, rtol=2e-6)
    np.testing.assert_allclose(encoder.cost(q, 0.15, weight=4), direct * 4, rtol=2e-6)


@pytest.mark.parametrize("dtype,rtol", [(jnp.float32, 1e-5), (jnp.float64, 1e-10)])
def test_single_and_batched_costs_match(encoder, motion, dtype, rtol):
    with jax.enable_x64(dtype == jnp.float64):
        model = jax.tree.map(lambda x: x.astype(dtype), encoder)
        q = jnp.asarray(np.stack([motion, motion + 0.02]), dtype=dtype)
        batched = model.cost(q, 0.15)
        individual = jnp.stack([model.cost(row, 0.15) for row in q])
        # Batch shape can select different convolution/reduction kernels. Float32
        # roundoff is amplified by the network and latent metric (about 2e-6 on
        # Linux CPU); keep the float64 check tight to catch batch coupling.
        np.testing.assert_allclose(batched, individual, rtol=rtol)


def test_padding_exact_matches_unpadded_prefix(encoder):
    x = jax.random.normal(jax.random.key(5), (1, 28, 53))
    padded = jnp.pad(x, ((0, 0), (0, 0), (0, 27)))
    mask = jnp.pad(jnp.ones((1, 1, 53)), ((0, 0), (0, 0), (0, 27)))
    # Both sides must use the runtime padding convention, not training-time pooling.
    a, _ = encoder.encode(x, padding_exact=True)
    b, _ = encoder.encode(padded, mask, padding_exact=True)
    np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)


def test_invalid_inputs(encoder):
    for q in [jnp.zeros((32, 6)), jnp.zeros((1, 7))]:
        with pytest.raises(ValueError):
            encoder.cost(q, 0.1)
    for dt in [0, -1, float("nan")]:
        with pytest.raises(ValueError):
            encoder.cost(jnp.zeros((32, 7)), dt)
    with pytest.raises(ValueError):
        encoder.cost(jnp.zeros((32, 7)), 0.1, weight=-1)
