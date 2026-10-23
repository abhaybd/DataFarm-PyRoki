"""Convert the pinned DATAFARM checkpoint into a pickle-free NumPy archive.

Run with `uv run python scripts/convert_checkpoint.py`. Downloads only the
SHA256-verified upstream artifact; torch is a development dependency, not runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

COMMIT = "fd403ea6eb16163261a13439fe9bbc10a980f50d"
SHA256 = "e31ef96b6ffc0af8ce890e8774b9b57d3fa6e91c48ba9125d1e3574219c30c5b"
URL = (
    f"https://raw.githubusercontent.com/SamratSahoo/DATAFARM/{COMMIT}"
    "/encoder/checkpoints/encoder.pt"
)
ROOT = Path(__file__).resolve().parents[1]


def read_verified_checkpoint(path: Path, expected_sha256: str | None = SHA256) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Checkpoint SHA256 mismatch: {digest}; expected {expected_sha256}")
    # The upstream artifact contains NumPy statistics alongside the torch state_dict.
    # Allow only the concrete NumPy constructors it needs; never unrestricted pickle.
    allowed = [
        np.ndarray,
        np.dtype,
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.dtypes.Float32DType,
        np.dtypes.Float64DType,
    ]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def convert(
    path: Path,
    output: Path,
    *,
    model_name: str = "droid-franka",
    joint_names: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8)),
    sample_rate_hz: float = 15.0,
    expected_sha256: str | None = SHA256,
) -> dict:
    blob = read_verified_checkpoint(path, expected_sha256)
    if int(blob["ch"]) != 4 * len(joint_names) or int(blob["n_joints"]) != len(joint_names):
        raise ValueError("joint_names must match the checkpoint's channel and joint counts")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("joint_names must be unique")
    arrays = {
        f"state_dict/{key}": tensor.detach().cpu().numpy()
        for key, tensor in blob["state_dict"].items()
    }
    for key in ("chan_mu", "chan_sd"):
        arrays[key] = np.asarray(blob[key])
    for suffix in ("mean", "cov", "precision"):
        key = f"reference_{suffix}"
        arrays[key] = np.asarray(blob[key] if key in blob else blob[f"droid_latent_{suffix}"])
    arrays["model_name"] = np.asarray(model_name)
    arrays["joint_names"] = np.asarray(joint_names)
    arrays["sample_rate_hz"] = np.asarray(sample_rate_hz)
    arrays["schema_version"] = np.asarray(1, dtype=np.int32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    provenance = {
        "model_name": model_name,
        "joint_names": list(joint_names),
        "sample_rate_hz": sample_rate_hz,
        "source_repository": "https://github.com/SamratSahoo/DATAFARM"
        if expected_sha256 == SHA256
        else None,
        "source_commit": COMMIT if expected_sha256 == SHA256 else None,
        "source_checkpoint": URL if expected_sha256 == SHA256 else path.name,
        "source_checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "converted_checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_model": f"https://github.com/SamratSahoo/DATAFARM/blob/{COMMIT}/encoder/model.py",
        "license": "MIT; Copyright (c) 2026 Samrat Sahoo; see DATAFARM-LICENSE.txt",
        "conversion": "Lossless extraction of state_dict tensors and reference statistics to NPZ",
        "training_metadata": {
            k: blob[k]
            for k in (
                "dataset",
                "dataset_codebase_version",
                "train_protocol",
                "n_droid_train",
                "n_droid_val",
                "n_droid_stats",
                "stats_max_segments",
                "hparams",
                "metrics",
                "kl_droid_mean",
                "maha2_droid_mean",
            )
            if k in blob
        },
    }
    output.with_name("provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "work/upstream/encoder.pt")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "src/datafarm_pyroki/assets/droid_encoder.npz"
    )
    parser.add_argument("--model-name", default="droid-franka")
    parser.add_argument("--joint-names", nargs="+", default=[f"joint{i}" for i in range(1, 8)])
    parser.add_argument("--sample-rate-hz", type=float, default=15.0)
    parser.add_argument("--sha256", default=SHA256, help="Expected source checksum")
    args = parser.parse_args()
    if not args.checkpoint.exists():
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            args.checkpoint.write_bytes(response.read())
    provenance = convert(
        args.checkpoint,
        args.output,
        model_name=args.model_name,
        joint_names=tuple(args.joint_names),
        sample_rate_hz=args.sample_rate_hz,
        expected_sha256=args.sha256,
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
