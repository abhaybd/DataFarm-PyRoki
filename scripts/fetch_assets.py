"""Prefetch the pinned Hugging Face models into the standard Hub cache.

Authenticate with `uv run hf auth login` or HF_TOKEN for private repositories.
Normal load_encoder()/load_joint_prior() calls fetch the same artifacts on demand.
"""

from __future__ import annotations

import argparse

from datafarm_pyroki.assets import resolve_model_asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("encoder", "posture-prior", "both"), default="both")
    parser.add_argument("--cache-dir", help="Override the standard HF_HUB_CACHE directory")
    parser.add_argument("--local-files-only", action="store_true", help="Require cached models")
    args = parser.parse_args()
    filenames = []
    if args.model in ("encoder", "both"):
        filenames.append("droid_encoder.npz")
    if args.model in ("posture-prior", "both"):
        filenames.append("droid_joint_prior.npz")
    for filename in filenames:
        path = resolve_model_asset(
            filename, cache_dir=args.cache_dir, local_files_only=args.local_files_only
        )
        print(f"Verified {filename}: {path}")


if __name__ == "__main__":
    main()
