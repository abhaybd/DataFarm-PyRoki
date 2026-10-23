"""Extract DATAFARM's pinned joint GMM; preserve its upstream NVIDIA license."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

COMMIT = "3a2e4d000339f7460f1989bde84d32b328ac92f9"
SHA256 = "f29cdecea3f0e24dd3b49ae1b889e85047aad2f5d9b376d39e959e8499e3c4e5"
URL = f"https://raw.githubusercontent.com/SamratSahoo/cuTAMP/{COMMIT}/cutamp/posture_ref.npz"
ROOT = Path(__file__).resolve().parents[1]


def convert(source: Path, output: Path):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Source joint prior checksum mismatch")
    with np.load(source, allow_pickle=False) as z:
        if str(z["model"]) != "gmm":
            raise ValueError("Expected the pinned Gaussian mixture prior")
        j = int(z["n_joints"])
        arrays = {
            "schema_version": np.int32(1),
            "model_name": np.asarray("droid-franka"),
            "joint_names": np.asarray([f"joint{i}" for i in range(1, j + 1)]),
            "weights": z["gmm_weights"],
            "means": z["gmm_means"],
            "precision_factors": z["gmm_prec_chol"],
            "location": z["gmm_mu"],
            "scale": z["gmm_scale"],
        }
        provenance = {
            "source": URL,
            "source_commit": COMMIT,
            "source_sha256": SHA256,
            "reference": str(z["provenance"]),
            "license": "NVIDIA License; see CUTAMP-LICENSE.txt",
            "conversion": "Lossless numeric GMM extraction; no refitting",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    provenance["converted_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name("joint_prior_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "work/upstream/posture_ref.npz")
    p.add_argument(
        "--output", type=Path, default=ROOT / "src/datafarm_pyroki/assets/droid_joint_prior.npz"
    )
    args = p.parse_args()
    if not args.source.exists():
        args.source.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            args.source.write_bytes(response.read())
    convert(args.source, args.output)
    # Ship the actual upstream notice with every converted distribution.
    license_path = ROOT / "licenses/CUTAMP.txt"
    args.output.with_name("CUTAMP-LICENSE.txt").write_bytes(license_path.read_bytes())
    print(args.output)


if __name__ == "__main__":
    main()
