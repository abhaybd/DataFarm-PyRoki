"""Fetch converted NPZs from supplied artifact URLs (including Hugging Face resolve URLs).

No host/repository is assumed. Expected checksums come from tracked provenance;
use explicit checksum overrides for a different artifact release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/datafarm_pyroki/assets"


def download(url: str, target: Path, expected_sha256: str):
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256:
        print(f"Already verified: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
        temporary = Path(output.name)
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            output.close()
            if digest.hexdigest() != expected_sha256:
                raise ValueError(f"SHA256 mismatch for {target.name}: {digest.hexdigest()}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Downloaded and verified: {target}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder-url")
    p.add_argument("--joint-prior-url")
    p.add_argument("--encoder-sha256")
    p.add_argument("--joint-prior-sha256")
    p.add_argument(
        "--destination", type=Path, default=Path(os.environ.get("DATAFARM_ASSET_DIR", ASSETS))
    )
    args = p.parse_args()
    if args.encoder_url is None and args.joint_prior_url is None:
        p.error("Provide --encoder-url and/or --joint-prior-url")
    encoder = json.loads((ASSETS / "provenance.json").read_text())
    prior = json.loads((ASSETS / "joint_prior_provenance.json").read_text())
    if args.encoder_url:
        download(
            args.encoder_url,
            args.destination / "droid_encoder.npz",
            args.encoder_sha256 or encoder["converted_checkpoint_sha256"],
        )
    if args.joint_prior_url:
        download(
            args.joint_prior_url,
            args.destination / "droid_joint_prior.npz",
            args.joint_prior_sha256 or prior["converted_sha256"],
        )
    # Keep provenance and redistribution notices alongside the downloaded assets.
    for name in (
        "provenance.json",
        "joint_prior_provenance.json",
        "DATAFARM-LICENSE.txt",
        "TIPTOP-LICENSE.txt",
        "CUTAMP-LICENSE.txt",
    ):
        source = ASSETS / name
        target = args.destination / name
        if source.resolve() != target.resolve():
            target.write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
