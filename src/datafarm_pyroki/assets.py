"""Pinned Hugging Face model downloads; checkpoint binaries stay outside the package."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError


@dataclass(frozen=True)
class HubAsset:
    repo_id: str
    revision: str
    sha256: str


MODEL_ASSETS = {
    "droid_encoder.npz": HubAsset(
        repo_id="abhaybd/DataFarm-Jax-MotionEncoder-Franka",
        revision="6fd59cbdfaf3387471f86fe31c3f1290b21dcbdc",
        sha256="ed6296980f448704c92e88cc91e9767a7dd4b758e82b93b80adc80178ee825a9",
    ),
    "droid_joint_prior.npz": HubAsset(
        repo_id="abhaybd/DataFarm-Jax-PosturePrior-Franka",
        revision="39338e248f79e3eba779ba17f0ff89a884c93ae4",
        sha256="3252f89dfb3a675d726ff6ef5fee3371244da6969d11a5bb7611425bebcf54e0",
    ),
}


def resolve_model_asset(
    filename: str,
    path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    token: str | bool | None = False,
    local_files_only: bool = False,
) -> Path:
    """Use an explicit file, otherwise download/check a pinned artifact in the HF cache.

    Public models download anonymously by default. Explicit paths
    bypass the Hub, its cache and its checksum contract, allowing custom models.
    local_files_only applies to Hub resolution and requires an already cached file.
    """
    if path is not None:
        return Path(path).expanduser()
    spec = MODEL_ASSETS[filename]
    try:
        resource = Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                filename=filename,
                revision=spec.revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )
        )
    except LocalEntryNotFoundError as error:
        raise FileNotFoundError(
            f"{spec.repo_id}/{filename} at revision {spec.revision} is not cached. "
            "Load once with network access, then use "
            "local_files_only=True or HF_HUB_OFFLINE=1 for offline loading."
        ) from error
    with resource.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != spec.sha256:
        raise ValueError(
            f"SHA256 mismatch for {spec.repo_id}/{filename}: expected {spec.sha256}, "
            f"got {actual}. Remove the corrupt entry from the HF cache and download again."
        )
    return resource


@contextmanager
def open_model_asset(filename: str, path: str | Path | None = None, **hub_options):
    """Open a resolved checkpoint without consulting source-tree artifact directories."""
    resource = resolve_model_asset(filename, path, **hub_options)
    with resource.open("rb") as stream:
        yield stream
