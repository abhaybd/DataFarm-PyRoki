"""Original DATAFARM artifacts and lossless JAX exports from the same learned parameters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

UPSTREAM_COMMIT = "fd403ea6eb16163261a13439fe9bbc10a980f50d"


def save_torch(path: Path, blob):
    import torch

    tmp = path.with_suffix(".tmp.pt")
    torch.save(blob, tmp)
    tmp.replace(path)


def export_encoder(blob: dict, path: Path):
    arrays = {f"state_dict/{k}": v.detach().cpu().numpy() for k, v in blob["state_dict"].items()}
    arrays.update(
        schema_version=np.int32(1),
        model_name=np.asarray(blob["model_name"]),
        joint_names=np.asarray(blob["joint_names"]),
        sample_rate_hz=np.asarray(blob["source_rate"]),
        chan_mu=np.asarray(blob["chan_mu"]),
        chan_sd=np.asarray(blob["chan_sd"]),
    )
    for suffix in ("mean", "cov", "precision"):
        arrays[f"reference_{suffix}"] = np.asarray(blob[f"droid_latent_{suffix}"])
    np.savez_compressed(path, **arrays)


def export_posture(blob: dict, path: Path):
    np.savez_compressed(
        path,
        schema_version=np.int32(1),
        model_name=blob["model_name"],
        joint_names=blob["joint_names"],
        weights=blob["gmm_weights"],
        means=blob["gmm_means"],
        precision_factors=blob["gmm_prec_chol"],
        location=blob["gmm_mu"],
        scale=blob["gmm_scale"],
    )


def write_checksums(root: Path):
    files = ["encoder.pt", "encoder.npz", "posture_ref.npz", "posture_prior.npz"]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    (root / "checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")
