import hashlib
import json
import runpy
import sys
from importlib.resources import files
from unittest.mock import Mock

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError, RepositoryNotFoundError

from datafarm_pyroki import assets as model_assets
from datafarm_pyroki import load_encoder, load_joint_prior


@pytest.mark.parametrize(
    "loader,filename,checkpoint_fixture",
    [
        (load_encoder, "droid_encoder.npz", "encoder_checkpoint"),
        (load_joint_prior, "droid_joint_prior.npz", "joint_prior_checkpoint"),
    ],
)
def test_loaders_use_pinned_hub_and_forward_options(
    monkeypatch, tmp_path, request, loader, filename, checkpoint_fixture
):
    checkpoint = request.getfixturevalue(checkpoint_fixture)
    download = Mock(return_value=str(checkpoint))
    monkeypatch.setattr(model_assets, "hf_hub_download", download)
    # The former source-directory override must not mask Hub loading.
    monkeypatch.setenv("DATAFARM_ASSET_DIR", str(tmp_path / "obsolete"))
    loaded = loader(cache_dir=tmp_path / "cache", token="test-token", local_files_only=True)
    spec = model_assets.MODEL_ASSETS[filename]
    download.assert_called_once_with(
        repo_id=spec.repo_id,
        filename=filename,
        revision=spec.revision,
        cache_dir=tmp_path / "cache",
        token="test-token",
        local_files_only=True,
    )
    if loader is load_encoder:
        assert loaded.spec.n_joints == 7
    else:
        assert len(loaded.joint_names) == 7
    # Calling without any options also resolves through the Hub.
    download.reset_mock()
    loader()
    assert download.call_args.kwargs["token"] is None
    assert download.call_args.kwargs["local_files_only"] is False


@pytest.mark.parametrize(
    "loader,checkpoint_fixture",
    [(load_encoder, "encoder_checkpoint"), (load_joint_prior, "joint_prior_checkpoint")],
)
def test_explicit_checkpoint_bypasses_hub(monkeypatch, request, loader, checkpoint_fixture):
    checkpoint = request.getfixturevalue(checkpoint_fixture)
    download = Mock(side_effect=AssertionError("Hub called"))
    monkeypatch.setattr(model_assets, "hf_hub_download", download)
    loader(checkpoint, local_files_only=True)
    download.assert_not_called()


def test_hub_integrity_and_pins_match_provenance(monkeypatch, tmp_path):
    package = files("datafarm_pyroki")
    for filename, provenance_file, checksum_key in (
        ("droid_encoder.npz", "provenance.json", "converted_checkpoint_sha256"),
        ("droid_joint_prior.npz", "joint_prior_provenance.json", "converted_sha256"),
    ):
        metadata = json.loads(package.joinpath(f"assets/{provenance_file}").read_text())
        spec = model_assets.MODEL_ASSETS[filename]
        assert spec.sha256 == metadata[checksum_key]
        assert len(spec.revision) == 40
        corrupt = tmp_path / filename
        corrupt.write_bytes(b"invalid checkpoint")
        monkeypatch.setattr(model_assets, "hf_hub_download", Mock(return_value=str(corrupt)))
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            model_assets.resolve_model_asset(filename)
        assert hashlib.sha256(corrupt.read_bytes()).hexdigest() != spec.sha256


def test_missing_explicit_file_does_not_fall_back(monkeypatch, tmp_path):
    download = Mock(side_effect=AssertionError("Hub called"))
    monkeypatch.setattr(model_assets, "hf_hub_download", download)
    with pytest.raises(FileNotFoundError):
        load_encoder(tmp_path / "missing.npz")
    download.assert_not_called()


def test_auth_and_offline_failures_are_actionable(monkeypatch):
    monkeypatch.setattr(
        model_assets,
        "hf_hub_download",
        Mock(
            side_effect=RepositoryNotFoundError(
                "private repo", response=Mock(headers={}, request=None)
            )
        ),
    )
    with pytest.raises(RuntimeError, match="hf auth login"):
        load_encoder()
    monkeypatch.setattr(
        model_assets, "hf_hub_download", Mock(side_effect=LocalEntryNotFoundError("not cached"))
    )
    with pytest.raises(FileNotFoundError, match="not cached"):
        load_joint_prior(local_files_only=True)


def test_prefetch_uses_same_resolver(monkeypatch, tmp_path, capsys):
    from pathlib import Path

    resolve = Mock(return_value=tmp_path / "model.npz")
    monkeypatch.setattr(model_assets, "resolve_model_asset", resolve)
    monkeypatch.setattr(
        sys, "argv", ["fetch_assets.py", "--cache-dir", str(tmp_path), "--local-files-only"]
    )
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/fetch_assets.py"), run_name="__main__"
    )
    assert [call.args[0] for call in resolve.call_args_list] == [
        "droid_encoder.npz",
        "droid_joint_prior.npz",
    ]
    assert all(
        call.kwargs == {"cache_dir": str(tmp_path), "local_files_only": True}
        for call in resolve.call_args_list
    )
    assert "Verified droid_encoder.npz" in capsys.readouterr().out
