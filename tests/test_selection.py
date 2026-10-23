import json
from importlib.resources import files

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from datafarm_pyroki import (
    JointMap,
    JointPositionPrior,
    NoFeasibleIKSolutions,
    load_joint_prior,
    rank_ik_solutions,
    select_ik_solution,
)


def small_prior():
    return JointPositionPrior.from_parameters(
        [0.3, 0.7],
        [[-1.0, 0.5], [1.0, -0.5]],
        [[[0.3, 0.1], [0.1, 0.7]], [[0.6, -0.2], [-0.2, 0.4]]],
        joint_names=("a", "b"),
        name="test-two-axis",
        location=[0.2, -0.3],
        scale=[2.0, 0.5],
    )


def test_full_nll_in_original_coordinates_and_gradient():
    prior = small_prior()
    q = np.array([[0.0, 0.0], [1.0, -0.4], [-2.0, 0.7]], np.float32)
    z = (q - np.asarray(prior.location)) / np.asarray(prior.scale)
    terms = []
    for weight, mean, factor in zip(prior.weights, prior.means, prior.precision_factors):
        p = np.asarray(factor) @ np.asarray(factor).T
        terms.append(
            np.log(weight) + multivariate_normal.logpdf(z, mean=mean, cov=np.linalg.inv(p))
        )
    expected = -logsumexp(np.stack(terms, -1), axis=-1) + np.log(np.asarray(prior.scale)).sum()
    actual = jax.jit(prior.nll)(jnp.asarray(q))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    grad = jax.grad(lambda x: prior.nll(x).sum())(jnp.asarray(q))
    for row, col in [(0, 0), (1, 1)]:
        step = np.zeros_like(q)
        step[row, col] = 1e-3
        difference = (
            np.asarray(prior.nll(q + step)).sum() - np.asarray(prior.nll(q - step)).sum()
        ) / 2e-3
        np.testing.assert_allclose(grad[row, col], difference, atol=2e-3, rtol=2e-3)


def test_bundled_prior_matches_upstream_scoring_formula():
    prior = load_joint_prior()
    assert prior.means.shape == (16, 7)
    q = np.random.default_rng(5).normal(size=(6, 7)).astype(np.float32)
    factor = np.asarray(prior.precision_factors)
    standardized = (q - prior.location) / prior.scale
    y = np.einsum("bkj,kjl->bkl", np.asarray(standardized)[:, None] - prior.means, factor)
    log_component = (
        np.log(prior.weights)
        + np.log(np.diagonal(factor, axis1=1, axis2=2)).sum(-1)
        - 0.5 * (y * y).sum(-1)
    )
    upstream_score = -logsumexp(log_component, axis=-1)
    constant = 7 / 2 * np.log(2 * np.pi) + np.log(prior.scale).sum()
    np.testing.assert_allclose(
        prior.nll(jnp.asarray(q)), upstream_score + constant, atol=2e-4, rtol=2e-6
    )
    provenance = files("datafarm_pyroki").joinpath("assets/joint_prior_provenance.json")
    assert "NVIDIA" in json.loads(provenance.read_text())["license"]


def test_rank_mask_reorder_bounds_and_batched_selection():
    prior = small_prior()
    mapping = JointMap.from_names(("b", "extra", "a"), prior.joint_names)
    # Same valid q twice: a tie must preserve its original index. The very low NLL
    # candidate is explicitly infeasible; invalid extra robot coordinates also count.
    q = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.55, 0.0, 2.2], [0.0, jnp.nan, 0.0]])
    mask = jnp.array([True, True, False, True])
    ranked = jax.jit(lambda q: rank_ik_solutions(prior, q, joint_map=mapping, feasible=mask))(q)
    np.testing.assert_array_equal(ranked.indices[:2], [0, 1])
    assert np.isinf(ranked.nll[2:]).all()
    best = select_ik_solution(
        prior, jnp.stack([q, q]), joint_map=mapping, feasible=jnp.stack([mask, mask])
    )
    np.testing.assert_array_equal(best.index, [0, 0])
    np.testing.assert_array_equal(best.positions, np.zeros((2, 3)))
    with pytest.raises(NoFeasibleIKSolutions):
        select_ik_solution(prior, q, joint_map=mapping, feasible=jnp.zeros(4, bool))
    with pytest.raises(NoFeasibleIKSolutions):
        select_ik_solution(
            prior,
            q,
            joint_map=mapping,
            feasible=mask,
            lower_limits=jnp.ones(3),
            upper_limits=jnp.ones(3) * 2,
        )


def test_custom_prior_roundtrip_and_reject_invalid_parameters(tmp_path):
    prior = small_prior()
    path = tmp_path / "custom.npz"
    prior.save(path)
    loaded = load_joint_prior(path)
    assert loaded.joint_names == ("a", "b")
    np.testing.assert_allclose(loaded.nll(jnp.ones(2)), prior.nll(jnp.ones(2)))
    with pytest.raises(ValueError, match="sum to one"):
        JointPositionPrior.from_parameters(
            [2.0], [[0.0]], [[[1.0]]], joint_names=("a",), name="bad"
        )
    with pytest.raises(np.linalg.LinAlgError):
        JointPositionPrior.from_parameters(
            [1.0], [[0.0]], [[[-1.0]]], joint_names=("a",), name="bad"
        )
