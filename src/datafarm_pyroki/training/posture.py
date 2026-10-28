"""Fit independent arm GMMs; scikit-learn for fitting, original arrays for PyTorch scoring."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from .artifacts import export_posture
from .data import JOINT_NAMES, Corpus, atomic_json
from .features import COMMON_RATE


def sample_nonidle(
    corpus: Corpus, arm: str, split: str, frames: int, episodes_per_source: int, seed: int
):
    rng = np.random.default_rng(seed)
    samples, counts = [], {}
    for source in ("abc", "molmoact2"):
        eligible = corpus.indices(split, source)
        chosen = rng.choice(eligible, min(len(eligible), episodes_per_source), replace=False)
        moving = []
        for i in chosen:
            q = np.asarray(corpus.joint(i, arm), np.float64)
            mask = np.linalg.norm(np.gradient(q, 1 / COMMON_RATE, axis=0), axis=1) > 0.05
            if mask.any():
                moving.append(q[mask])
        if not moving:
            raise ValueError(f"{source}/{arm}/{split}: no non-idle frames")
        pool = np.concatenate(moving)
        n = frames // 2
        if len(pool) < n:
            raise ValueError(f"{source}: need {n} non-idle frames, found {len(pool)}")
        samples.append(pool[rng.choice(len(pool), n, replace=False)])
        counts[source] = {
            "episodes": len(chosen),
            "available_nonidle_frames": len(pool),
            "sampled_frames": n,
        }
    return np.concatenate(samples), counts


def fit_posture(
    corpus: Corpus,
    arm: str,
    out: Path,
    *,
    components=16,
    frames=120000,
    episodes_per_source=2000,
    seed=0,
    validation_frames=12000,
):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits

    if frames % 2 or validation_frames % 2:
        raise ValueError("Frame budgets must be even for an exact 50/50 mixture")
    q, counts = sample_nonidle(corpus, arm, "train", frames, episodes_per_source, seed)
    mu, scale = q.mean(0), np.maximum(q.std(0), 1e-6)
    print(f"[posture] {arm}: fitting K={components} to {len(q)} non-idle frames", flush=True)
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = GaussianMixture(
            n_components=components,
            covariance_type="full",
            random_state=seed,
            reg_covar=1e-4,
            max_iter=200,
        ).fit((q - mu) / scale)
    validation, val_counts = sample_nonidle(
        corpus, arm, "val", validation_frames, len(corpus.indices("val")), seed + 1
    )
    val_nll = -model.score_samples((validation - mu) / scale) + np.log(scale).sum()
    report = {
        "manifest_sha256": corpus.digest,
        "arm": arm,
        "components": components,
        "training_sampling": counts,
        "validation_sampling": val_counts,
        "converged": bool(model.converged_),
        "iterations": int(model.n_iter_),
        "heldout_nll": {
            source: float(values.mean())
            for source, values in zip(("abc", "molmoact2"), np.split(val_nll, 2), strict=True)
        },
    }
    names = JOINT_NAMES[:6] if arm == "left" else JOINT_NAMES[6:]
    blob = dict(
        model="gmm",
        model_name=f"yam-{arm}-abc-molmoact2",
        n_joints=6,
        joint_names=np.asarray(names),
        sample_rate_hz=COMMON_RATE,
        n_frames=len(q),
        provenance=json.dumps(report),
        gmm_weights=model.weights_,
        gmm_means=model.means_,
        gmm_prec_chol=model.precisions_cholesky_,
        gmm_mu=mu,
        gmm_scale=scale,
        gmm_k=components,
    )
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "posture_ref.npz", **blob)
    export_posture(blob, out / "posture_prior.npz")
    atomic_json(out / "posture_report.json", report)
    return report
