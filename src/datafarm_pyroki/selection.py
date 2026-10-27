"""Joint-space density scoring and feasible IK branch selection, independent of PyRoki."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from .assets import open_model_asset
from .embodiment import JointMap


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["weights", "means", "precision_factors", "location", "scale"],
    meta_fields=["name", "joint_names"],
)
@dataclass(frozen=True)
class JointPositionPrior:
    """A full-covariance GMM in standardized training joint coordinates.

    Each precision factor C obeys precision = C @ C.T (upper OR lower triangular).
    Means/factors describe z = (q - location)/scale; nll includes the change of units
    back to q. Model coordinates are explicit and independent of robot DOF count.
    """

    weights: jax.Array
    means: jax.Array
    precision_factors: jax.Array
    location: jax.Array
    scale: jax.Array
    name: str
    joint_names: tuple[str, ...]

    def nll(self, q: jax.Array) -> jax.Array:
        """Full negative log density in input joint units; supports arbitrary batch axes."""
        q = jnp.asarray(q)
        if q.ndim < 1 or q.shape[-1] != len(self.joint_names):
            raise ValueError("Joint positions must use the prior's training joint order")
        z = (q - self.location) / self.scale
        error = z[..., None, :] - self.means
        whitened = jnp.einsum("...kj,kjl->...kl", error, self.precision_factors)
        log_components = (
            jnp.log(self.weights)
            + jnp.log(jnp.diagonal(self.precision_factors, axis1=-2, axis2=-1)).sum(-1)
            - 0.5 * jnp.sum(whitened**2, axis=-1)
        )
        constant = 0.5 * len(self.joint_names) * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale).sum()
        return -logsumexp(log_components, axis=-1) + constant

    @classmethod
    def from_parameters(
        cls,
        weights,
        means,
        covariances,
        *,
        joint_names: Sequence[str],
        name: str,
        location=None,
        scale=None,
    ) -> JointPositionPrior:
        """Construct from a fitted GMM; covariances are in standardized coordinates.

        For a GMM fit directly to raw q, leave location/scale at their defaults.
        This does not fit/retrain a model or infer compatibility from its dimension.
        """
        means = np.asarray(means, dtype=np.float64)
        if means.ndim != 2:
            raise ValueError("means must have shape [components,joints]")
        k, j = means.shape
        covariance = np.asarray(covariances, dtype=np.float64)
        if covariance.shape != (k, j, j) or not np.isfinite(covariance).all():
            raise ValueError("covariances must be finite [components,joints,joints]")
        if not np.allclose(covariance, covariance.swapaxes(-1, -2), atol=1e-8):
            raise ValueError("Covariances must be symmetric")
        np.linalg.cholesky(covariance)  # reject indefinite covariances before inversion
        factors = np.linalg.cholesky(np.linalg.inv(covariance))
        return _validated_prior(
            weights,
            means,
            factors,
            np.zeros(j) if location is None else location,
            np.ones(j) if scale is None else scale,
            name,
            tuple(joint_names),
        )

    def save(self, path: str | Path) -> None:
        """Save a portable, pickle-free artifact for this model."""
        np.savez_compressed(
            path,
            schema_version=np.int32(1),
            model_name=np.asarray(self.name),
            joint_names=np.asarray(self.joint_names),
            weights=np.asarray(self.weights),
            means=np.asarray(self.means),
            precision_factors=np.asarray(self.precision_factors),
            location=np.asarray(self.location),
            scale=np.asarray(self.scale),
        )


def _validated_prior(weights, means, factors, location, scale, name, names):
    w, m, f, loc, s = map(np.asarray, (weights, means, factors, location, scale))
    j = len(names)
    k = len(w) if w.ndim == 1 else 0
    if not name or not j or len(set(names)) != j or any(not n for n in names):
        raise ValueError("A prior needs a name and unique ordered training joint names")
    if not k or m.shape != (k, j) or f.shape != (k, j, j) or loc.shape != (j,) or s.shape != (j,):
        raise ValueError("Inconsistent GMM parameter dimensions")
    if not all(np.isfinite(x).all() for x in (w, m, f, loc, s)):
        raise ValueError("GMM parameters must be finite")
    if np.any(w <= 0) or not np.isclose(w.sum(), 1.0, atol=1e-6) or np.any(s <= 0):
        raise ValueError("Mixture weights must be positive and sum to one; scales must be positive")
    if np.any(np.diagonal(f, axis1=-2, axis2=-1) <= 0):
        raise ValueError("Precision factors must have positive diagonals")
    for factor in f:
        if not (np.allclose(factor, np.tril(factor)) or np.allclose(factor, np.triu(factor))):
            raise ValueError("Precision factors must be triangular")
    return JointPositionPrior(
        *(jnp.asarray(x, dtype=jnp.float32) for x in (w, m, f, loc, s)), str(name), tuple(names)
    )


def load_joint_prior(
    path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    token: str | bool | None = False,
    local_files_only: bool = False,
) -> JointPositionPrior:
    """Load the pinned HF posture GMM, or an explicit pickle-free NPZ for another model.

    Public weights download anonymously. cache_dir overrides the Hub cache;
    local_files_only requires the pinned checkpoint to have been cached already.
    """
    with open_model_asset(
        "droid_joint_prior.npz",
        path,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    ) as stream:
        with np.load(stream, allow_pickle=False) as z:
            if int(z["schema_version"]) != 1:
                raise ValueError("Unsupported joint prior schema")
            return _validated_prior(
                z["weights"],
                z["means"],
                z["precision_factors"],
                z["location"],
                z["scale"],
                str(z["model_name"]),
                tuple(z["joint_names"].tolist()),
            )


class IKRanking(NamedTuple):
    indices: jax.Array  # increasing NLL, stable for ties; invalid candidates last
    nll: jax.Array  # scores in ORIGINAL candidate order; invalid -> +inf
    feasible: jax.Array


class IKSelection(NamedTuple):
    positions: jax.Array  # selected complete ROBOT configurations, not only model joints
    index: jax.Array
    nll: jax.Array


class NoFeasibleIKSolutions(ValueError):
    """At least one query has no finite, feasible IK candidate."""


def rank_ik_solutions(
    prior: JointPositionPrior,
    candidates: jax.Array,
    *,
    joint_map: JointMap,
    feasible: jax.Array,
    lower_limits=None,
    upper_limits=None,
) -> IKRanking:
    """JIT-compatible ranking of [..., candidates, robot_joints].

    The caller's mask must encode IK pose success and collision/task feasibility.
    This utility adds finite-coordinate checks and optional robot-order joint limits;
    it never treats a low NLL as evidence of kinematic feasibility.
    """
    q = jnp.asarray(candidates)
    if q.ndim < 2 or q.shape[-2] == 0:
        raise ValueError("Expected at least one IK candidate [...,candidates,robot_joints]")
    if joint_map.model_joint_names != prior.joint_names:
        raise ValueError("JointMap does not match the prior's joint order")
    valid = jnp.asarray(feasible, dtype=bool)
    if valid.shape != q.shape[:-1]:
        raise ValueError("feasible must have shape [...,candidates]")
    valid = valid & jnp.isfinite(q).all(-1)
    if (lower_limits is None) != (upper_limits is None):
        raise ValueError("Supply both lower_limits and upper_limits")
    if lower_limits is not None:
        lower, upper = jnp.asarray(lower_limits), jnp.asarray(upper_limits)
        if lower.shape != (q.shape[-1],) or upper.shape != lower.shape:
            raise ValueError("Limits must contain one value per robot joint")
        valid = valid & ((q >= lower) & (q <= upper)).all(-1)
    safe_q = jnp.where(jnp.isfinite(q), q, 0.0)
    score = prior.nll(joint_map.select(safe_q))
    valid = valid & jnp.isfinite(score)
    score = jnp.where(valid, score, jnp.inf)
    return IKRanking(jnp.argsort(score, axis=-1, stable=True), score, valid)


def select_ik_solution(
    prior: JointPositionPrior,
    candidates: jax.Array,
    *,
    joint_map: JointMap,
    feasible: jax.Array,
    lower_limits=None,
    upper_limits=None,
) -> IKSelection:
    """Host-side selection, with a clear error instead of choosing an invalid candidate.

    Supports single and batched queries. Use rank_ik_solutions inside a JIT and
    explicitly handle the all-invalid mask there; this convenience wrapper raises.
    """
    q = jnp.asarray(candidates)
    ranking = rank_ik_solutions(
        prior,
        q,
        joint_map=joint_map,
        feasible=feasible,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )
    has_candidate = np.asarray(ranking.feasible.any(-1))
    if not has_candidate.all():
        bad = np.argwhere(~has_candidate).tolist() if has_candidate.ndim else [()]
        raise NoFeasibleIKSolutions(f"No feasible IK candidates for queries {bad}")
    index = ranking.indices[..., 0]
    selected = jnp.take_along_axis(q, index[..., None, None], axis=-2)[..., 0, :]
    nll = jnp.take_along_axis(ranking.nll, index[..., None], axis=-1)[..., 0]
    return IKSelection(selected, index, nll)
