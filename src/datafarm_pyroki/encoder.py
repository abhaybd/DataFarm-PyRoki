"""Frozen DATAFARM encoder inference in JAX.

Architecture ported from Samrat Sahoo's MIT-licensed DATAFARM encoder/model.py.
See licenses/DATAFARM.txt and assets/provenance.json. No PyTorch at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from .assets import open_model_asset
from .embodiment import EncoderSpec

RATE_HZ = 15.0
FILTERS = ((3, 1), (7, 1), (15, 1), (7, 2), (15, 4), (15, 8))


def sample_count(horizon: int, source_dt: float, rate_hz: float = RATE_HZ) -> int:
    """Nominal model-rate output size; dt/horizon/rate are static under JIT."""
    if horizon < 2:
        raise ValueError("A motion leg needs at least two waypoints")
    if not np.isfinite(source_dt) or source_dt <= 0:
        raise ValueError("source_dt must be a positive, finite Python scalar")
    return max(2, int(round((horizon - 1) * float(source_dt) * rate_hz)) + 1)


def time_derivative(q: Array, dt: float = 1.0 / RATE_HZ) -> Array:
    """Central differences internally, one-sided differences at the edges (axis -2)."""
    if q.shape[-2] < 2:
        raise ValueError("At least two frames are required for finite differences")
    return jnp.concatenate(
        (
            (q[..., 1:2, :] - q[..., :1, :]) / dt,
            (q[..., 2:, :] - q[..., :-2, :]) / (2.0 * dt),
            (q[..., -1:, :] - q[..., -2:-1, :]) / dt,
        ),
        axis=-2,
    )


def resample_positions(q: Array, count: int) -> Array:
    """Endpoint-aligned linear interpolation along axis -2 (PyTorch align_corners=True)."""
    if q.shape[-2] == count:
        return q
    index = jnp.linspace(0.0, q.shape[-2] - 1, count, dtype=q.dtype)
    lo = jnp.floor(index).astype(jnp.int32)
    hi = jnp.minimum(lo + 1, q.shape[-2] - 1)
    fraction = (index - lo)[..., None]
    return q[..., lo, :] * (1.0 - fraction) + q[..., hi, :] * fraction


def _moments(x: Array, mask: Array) -> tuple[Array, Array]:
    count = jnp.maximum(mask.sum(-1), 1.0)
    mean = (x * mask).sum(-1) / count
    variance = (((x - mean[..., None]) ** 2) * mask).sum(-1) / count
    return mean, jnp.sqrt(jnp.maximum(variance, 1e-8))


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "state_dict",
        "chan_mu",
        "chan_sd",
        "latent_mean",
        "latent_precision",
        "precision_cholesky",
    ],
    meta_fields=["spec"],
)
@dataclass(frozen=True)
class MotionEncoder:
    """Pretrained parameters and reference distribution, all ordinary JAX pytrees.

    `encode` accepts standardized [B,4*J,T] features; trajectory methods accept
    [T,J] or [B,T,J] joint positions in radians, in spec.joint_names order.
    Architecture width, latent dimension, and sampling rate come from the artifact.
    The default reference model is specific to DROID Franka; other robots need their own weights.
    """

    state_dict: dict[str, Array]
    chan_mu: Array
    chan_sd: Array
    latent_mean: Array
    latent_precision: Array
    precision_cholesky: Array
    spec: EncoderSpec

    def _linear(self, x: Array, name: str) -> Array:
        return x @ self.state_dict[f"{name}.weight"].T + self.state_dict[f"{name}.bias"]

    def _conv(self, x: Array, name: str, *, stride: int = 1, dilation: int = 1) -> Array:
        kernel = self.state_dict[f"{name}.weight"].astype(x.dtype)
        pad = dilation * (kernel.shape[-1] - 1) // 2
        y = lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(stride,),
            padding=((pad, pad),),
            rhs_dilation=(dilation,),
            dimension_numbers=("NCH", "OIH", "NCH"),
            precision=lax.Precision.HIGHEST,
        )
        return y + self.state_dict[f"{name}.bias"][None, :, None]

    def _batch_norm(self, x: Array, name: str) -> Array:
        p = self.state_dict
        mean = p[f"{name}.running_mean"][None, :, None]
        var = p[f"{name}.running_var"][None, :, None]
        scale = p[f"{name}.weight"][None, :, None]
        bias = p[f"{name}.bias"][None, :, None]
        return (x - mean) * lax.rsqrt(var + 1e-5) * scale + bias

    def _layer_norm(self, x: Array, name: str) -> Array:
        mean = x.mean(-1, keepdims=True)
        variance = ((x - mean) ** 2).mean(-1, keepdims=True)
        return (x - mean) * lax.rsqrt(variance + 1e-5) * self.state_dict[
            f"{name}.weight"
        ] + self.state_dict[f"{name}.bias"]

    def encode(
        self, x: Array, mask: Array | None = None, *, padding_exact: bool = False
    ) -> tuple[Array, Array]:
        """Return latent (mean, log variance), each [B,D].

        Default reproduces the released training encoder's eval-mode forward pass.
        `padding_exact=True` implements the runtime prefix-preserving variant for
        padded/soft-masked retiming inputs. In that mode x must be zero past mask.
        This distinction matters: ordinary strided convolutions propagate biases
        through padded frames. Do not interchange the modes when checking parity.
        Dropout is disabled; BatchNorm uses stored running statistics; GELU is exact.
        """
        x = jnp.asarray(x)
        if x.ndim != 3 or x.shape[1] != 4 * self.spec.n_joints or x.shape[2] < 2:
            raise ValueError("Expected standardized features [batch,4*n_joints,time>=2]")
        if mask is None:
            mask = jnp.ones((x.shape[0], 1, x.shape[-1]), dtype=x.dtype)
        if mask.shape != (x.shape[0], 1, x.shape[-1]):
            raise ValueError("mask must have shape [batch,1,time]")
        pooled = []
        for i, (_, dilation) in enumerate(FILTERS):
            band = jnp.abs(
                self._batch_norm(self._conv(x, f"fb.{i}", dilation=dilation), f"fbn.{i}")
            )
            pooled.extend(_moments(band, mask))
        h, m = x, mask
        for conv, bn in ((0, 1), (3, 4), (6, 7)):
            h = self._conv(h, f"t.{conv}", stride=2)
            if padding_exact:
                m = m[:, :, ::2][:, :, : h.shape[-1]]
                h = h * m
            h = self._batch_norm(h, f"t.{bn}")
            if padding_exact:
                h = h * m
            h = jax.nn.gelu(h, approximate=False)
            if padding_exact:
                h = h * m
        if not padding_exact:
            m = mask[:, :, ::8][:, :, : h.shape[-1]]
        mean, std = _moments(h, m)
        max_mask = (m > 0.5).astype(h.dtype) if padding_exact else m
        maximum = (h + (max_mask - 1.0) * 1e9).max(-1)
        h = jnp.concatenate([*pooled, mean, std, maximum], axis=-1)
        h = jax.nn.gelu(self._layer_norm(self._linear(h, "fc.0"), "fc.1"), approximate=False)
        h = jax.nn.gelu(self._layer_norm(self._linear(h, "fc.4"), "fc.5"), approximate=False)
        return tuple(jnp.split(self._linear(h, "to_lat"), 2, axis=-1))

    def predict_fingerprint(self, latent: Array) -> Array:
        """Training auxiliary head; not needed by the planning cost."""
        return self._linear(jax.nn.gelu(self._linear(latent, "aux.0"), approximate=False), "aux.2")

    def features_at_model_rate(self, q: Array) -> Array:
        """Positions [B,T,J] already sampled at spec.sample_rate_hz -> [B,4*J,T]."""
        q = jnp.asarray(q)
        if q.ndim != 3 or q.shape[-1] != self.spec.n_joints:
            raise ValueError("Expected positions [batch,time,n_joints] in model joint order")
        if not jnp.issubdtype(q.dtype, jnp.floating):
            q = q.astype(jnp.float32)
        v = time_derivative(q, 1.0 / self.spec.sample_rate_hz)
        a = time_derivative(v, 1.0 / self.spec.sample_rate_hz)
        jerk = time_derivative(a, 1.0 / self.spec.sample_rate_hz)
        features = jnp.concatenate((q, v, a, jerk), axis=-1)
        return ((features - self.chan_mu) / self.chan_sd).transpose(0, 2, 1)

    def preprocess(self, q: Array, source_dt: float) -> Array:
        """Resample a uniformly timed leg at the model rate and build derivative features.

        `source_dt` is the waypoint timestep, not the controller playback timestep.
        It must be static under JIT: close over it or mark it static_argnames.
        """
        q = jnp.asarray(q)
        if q.ndim == 2:
            q = q[None]
        if q.ndim != 3 or q.shape[-1] != self.spec.n_joints:
            raise ValueError("Expected joint positions [time,n_joints] or [batch,time,n_joints]")
        if not jnp.issubdtype(q.dtype, jnp.floating):
            q = q.astype(jnp.float32)
        return self.features_at_model_rate(
            resample_positions(q, sample_count(q.shape[-2], source_dt, self.spec.sample_rate_hz))
        )

    def latent(self, q: Array, source_dt: float) -> Array:
        """Deterministic trajectory embedding. No latent sampling during planning."""
        mean, _ = self.encode(self.preprocess(q, source_dt))
        return mean[0] if q.ndim == 2 else mean

    def residual(self, q: Array, source_dt: float, weight: float = 1.0) -> Array:
        """Whitened latent error; its squared norm is weight * Mahalanobis².

        A scalar weight multiplies the COST, so the residual uses sqrt(weight).
        The lower Cholesky factor C satisfies precision = C @ C.T; row errors
        multiply C (not C.T). Freeze parameters while differentiating q.
        """
        if isinstance(weight, (float, int)) and (not np.isfinite(weight) or weight < 0):
            raise ValueError("weight must be nonnegative and finite")
        delta = self.latent(q, source_dt) - self.latent_mean
        return jnp.sqrt(weight) * (delta @ self.precision_cholesky)

    def cost(self, q: Array, source_dt: float, weight: float = 1.0) -> Array:
        """Return one scalar per trajectory, never one score per waypoint."""
        residual = self.residual(q, source_dt, weight)
        return jnp.sum(residual**2, axis=-1)


def load_encoder(path: str | Path | None = None) -> MotionEncoder:
    """Load an external NumPy checkpoint (pickle disabled), defaulting to local DROID weights."""
    with open_model_asset("droid_encoder.npz", path) as stream:
        with np.load(stream, allow_pickle=False) as blob:
            if int(blob["schema_version"]) != 1:
                raise ValueError("Unsupported encoder checkpoint schema")
            state = {
                key.removeprefix("state_dict/"): jnp.asarray(blob[key])
                for key in blob.files
                if key.startswith("state_dict/")
            }
            spec = EncoderSpec(
                str(blob["model_name"]),
                tuple(blob["joint_names"].tolist()),
                float(blob["sample_rate_hz"]),
            )
            width = 4 * spec.n_joints
            latent_dim = blob["reference_mean"].shape[0]
            if (
                blob["chan_mu"].shape != (width,)
                or blob["chan_sd"].shape != (width,)
                or state["fb.0.weight"].shape[1] != width
                or state["to_lat.weight"].shape[0] != 2 * latent_dim
            ):
                raise ValueError("Encoder dimensions do not match the model metadata")
            precision = np.asarray(blob["reference_precision"], dtype=np.float64)
            if precision.shape != (latent_dim, latent_dim) or not np.isfinite(precision).all():
                raise ValueError("Invalid reference precision matrix")
            chol = np.linalg.cholesky((precision + precision.T) / 2.0).astype(np.float32)
            if not np.all(np.isfinite(blob["chan_sd"]) & (blob["chan_sd"] > 0)):
                raise ValueError("Invalid input normalization scales")
            return MotionEncoder(
                state,
                jnp.asarray(blob["chan_mu"]),
                jnp.asarray(blob["chan_sd"]),
                jnp.asarray(blob["reference_mean"]),
                jnp.asarray(precision, dtype=jnp.float32),
                jnp.asarray(chol),
                spec,
            )
