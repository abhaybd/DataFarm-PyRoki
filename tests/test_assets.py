import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest

from datafarm_pyroki import load_encoder, load_joint_prior

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/datafarm_pyroki/assets"


def test_external_asset_directory_and_explicit_path_precedence(monkeypatch, tmp_path):
    # A configured external directory must not silently fall back to source-tree weights.
    monkeypatch.setenv("DATAFARM_ASSET_DIR", str(tmp_path))
    for loader in (load_encoder, load_joint_prior):
        with pytest.raises(FileNotFoundError, match="DATAFARM_ASSET_DIR"):
            loader()
    encoder = load_encoder(ASSETS / "droid_encoder.npz")
    prior = load_joint_prior(ASSETS / "droid_joint_prior.npz")
    for filename in ("droid_encoder.npz", "droid_joint_prior.npz"):
        (tmp_path / filename).write_bytes((ASSETS / filename).read_bytes())
    np.testing.assert_array_equal(load_encoder().latent_mean, encoder.latent_mean)
    np.testing.assert_array_equal(load_joint_prior().means, prior.means)


def test_fetch_verifies_checksum_and_preserves_existing_file_on_failure(tmp_path):
    download = runpy.run_path(str(ROOT / "scripts/fetch_assets.py"))["download"]
    source = tmp_path / "source.npz"
    target = tmp_path / "external/weights.npz"
    source.write_bytes(b"new artifact bytes")
    target.parent.mkdir()
    target.write_bytes(b"existing artifact bytes")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        download(source.as_uri(), target, "0" * 64)
    assert target.read_bytes() == b"existing artifact bytes"
    assert list(target.parent.iterdir()) == [target]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    download(source.as_uri(), target, digest)
    assert target.read_bytes() == source.read_bytes()
    # Matching cached weights need no network and may be used offline.
    download("https://invalid.invalid/unavailable", target, digest)
