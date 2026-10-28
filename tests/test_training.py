"""Training conventions, episode isolation, and original/JAX artifact compatibility."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

from datafarm_pyroki import load_encoder, load_joint_prior
from datafarm_pyroki.training.data import (
    JOINT_NAMES,
    SOURCES,
    Corpus,
    atomic_json,
    canonical_episode,
    choose_splits,
    manifest_hash,
    prepare,
    state_columns,
)
from datafarm_pyroki.training.features import COMMON_RATE, Traj, fingerprint, metric_series
from datafarm_pyroki.training.model import TrajectoryEncoder
from datafarm_pyroki.training.preprocess import (
    FeatureCache,
    fit_target_statistics,
    prepare_features,
    window_variants,
)
from datafarm_pyroki.training.train import TrainConfig, train_arm


def synthetic_corpus(root):
    rng = np.random.default_rng(8)
    positions, sources, ids, splits, lengths = [], [], [], [], []
    for source in SOURCES:
        for ep in range(6):
            length = 100 + ep * 5
            t = np.arange(length)[:, None] / 30
            q = 0.4 * np.sin(t * np.arange(1, 13)[None] + rng.uniform(-1, 1, (1, 12)))
            positions.append(q.astype(np.float32))
            sources.append(source)
            ids.append(ep)
            splits.append("train" if ep < 4 else "val")
            lengths.append(length)
    root.mkdir(parents=True)
    manifest = {
        "sample_rate_hz": 30.0,
        "joint_names": list(JOINT_NAMES),
        "sources": {k: {"repo": v["repo"], "revision": v["revision"]} for k, v in SOURCES.items()},
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "corpus.json", {"manifest_sha256": manifest_hash(manifest)})
    np.save(root / "positions.npy", np.concatenate(positions))
    np.savez(
        root / "index.npz",
        offsets=np.r_[0, np.cumsum(lengths)],
        sources=sources,
        episode_ids=ids,
        splits=splits,
    )
    return Corpus(root)


def test_explicit_joint_mapping_and_native_rate():
    for source, spec in SOURCES.items():
        names = list(reversed(spec["state_names"]))
        info = {"fps": 30, "features": {"observation.state": {"shape": [14], "names": names}}}
        actual = [names[i] for i in state_columns(info, source)]
        assert actual == [n for n in spec["state_names"] if "gripper" not in n]
        info["fps"] = 15
        with pytest.raises(ValueError, match="30"):
            state_columns(info, source)
    q = np.arange(90)[:, None] / COMMON_RATE * np.arange(1, 7)[None]
    np.testing.assert_allclose(
        metric_series(q)[:, 6:12], np.broadcast_to(np.arange(1, 7), (90, 6)), atol=1e-12
    )
    assert len(fingerprint(Traj("", "", q, 30, len(q)))) == 99


def test_uniform_episode_splits_are_disjoint_reproducible():
    rows = [{"episode_index": i, "length": 200 if i else 3} for i in range(100)]
    a = choose_splits(rows, 30, 10, seed=2)
    assert a == choose_splits(list(reversed(rows)), 30, 10, seed=2)
    assert len(a["train"]) == 30 and len(a["val"]) == 10
    assert not set(a["train"]) & set(a["val"])
    assert 0 not in a["train"] + a["val"]
    with pytest.raises(ValueError, match="eligible"):
        choose_splits(rows, 100, 10, 2)


def test_native_episode_assembly_rejects_missing_frames_and_clock_gaps():
    n = 90
    q = np.broadcast_to(np.arange(12) / 10, (n, 12))
    frame = np.arange(n)[::-1]
    actual, audit = canonical_episode(q[frame], frame / 30, frame, n)
    np.testing.assert_array_equal(actual, q.astype(np.float32))
    assert audit["dt_min"] == pytest.approx(1 / 30)
    with pytest.raises(ValueError, match="frames"):
        canonical_episode(q[:-1], np.arange(n - 1) / 30, np.arange(n - 1), n)
    times = np.arange(n) / 30
    times[40:] += 1
    with pytest.raises(ValueError, match="clock"):
        canonical_episode(q, times, np.arange(n), n)


def test_file_spanning_episodes_are_reassembled(tmp_path, monkeypatch):
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    import datafarm_pyroki.training.data as data

    raw = tmp_path / "raw"
    raw.mkdir()
    paths = ["data/chunk-000/file-000.parquet", "data/chunk-000/file-001.parquet"]
    rows = [
        {
            "episode_index": i,
            "length": 80,
            "data/chunk_index": 0,
            "data/file_index": 0 if i < 2 else 1,
            "dataset_from_index": i * 80,
        }
        for i in range(3)
    ]
    records = [
        {
            "observation.state": (np.arange(14) / 10).tolist(),
            "timestamp": j / 30,
            "frame_index": j,
            "episode_index": ep,
        }
        for ep in range(3)
        for j in range(80)
    ]
    for path, values in zip(paths, (records[:100], records[100:]), strict=True):
        pq.write_table(pa.Table.from_pylist(values), raw / Path(path).name, row_group_size=40)

    class LocalFS:
        def open(self, path, *args, **kwargs):
            return open(raw / Path(path).name, "rb")

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", LocalFS)
    monkeypatch.setattr(
        data,
        "_metadata",
        lambda root, source: (
            {"total_episodes": 3},
            (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12),
            rows,
            paths,
        ),
    )
    root = tmp_path / "corpus"
    prepare(root, train=2, val=1, seed=0, workers=1)
    corpus = Corpus(root)
    assert len(corpus.indices("train")) == 4 and len(corpus.indices("val")) == 2
    for i in range(6):
        assert corpus.joint(i, "left").shape == (80, 6)
        np.testing.assert_allclose(corpus.joint(i, "right")[0], np.arange(7, 13) / 10)


def test_paired_windows_and_training_only_imputation(tmp_path):
    corpus = synthetic_corpus(tmp_path / "corpus")
    left = corpus.joint(0, "left").copy()
    windows = list(window_variants(left, crops=6, seed=3, max_length=70))
    assert len(windows) == 7 and all(60 <= len(w) <= 70 for w in windows)
    cache = FeatureCache(
        prepare_features(corpus, "left", tmp_path / "features", crops=1, workers=1, max_length=70)
    )
    w = list(window_variants(left, crops=1, seed=0, max_length=70))[0]
    raw = np.asarray(list(fingerprint(Traj("", "", w, 30, len(w))).values()))
    m = cache.metadata
    raw = np.where(np.isfinite(raw), raw, m["feat_impute"])
    np.testing.assert_allclose(m["targets"][0], (raw - m["feat_mu"]) / m["feat_sd"], atol=1e-5)
    np.testing.assert_allclose(cache.series[0, : len(w)], metric_series(w), atol=1e-5)
    impute, mean, scale = fit_target_statistics(np.array([[np.nan, 1], [np.nan, 3]]))
    np.testing.assert_array_equal(impute, [0, 2])
    assert np.isfinite(mean).all() and np.all(scale > 0)
    assert not set(m["episodes"][m["splits"] == "train"]) & set(m["episodes"][m["splits"] == "val"])
    np.testing.assert_array_equal(corpus.joint(0, "left"), left)


def test_cpu_training_exports_load_in_torch_and_jax_and_resume(tmp_path):
    corpus = synthetic_corpus(tmp_path / "corpus")
    torch.set_num_threads(1)
    cfg = TrainConfig(
        epochs=1, steps=2, batch=4, crops=1, stats_segments=20, stats_batch=4, eval_batch=2
    )
    out = tmp_path / "left"
    args = (corpus, "left", cfg, "cpu", out, tmp_path / "features", 1, False, 2, 200, 4, 100)
    train_arm(*args)
    encoder = load_encoder(out / "encoder.npz")
    prior = load_joint_prior(out / "posture_prior.npz")
    assert encoder.spec.sample_rate_hz == 30
    assert encoder.spec.joint_names == tuple(JOINT_NAMES[:6]) == prior.joint_names
    blob = torch.load(out / "encoder.pt", map_location="cpu", weights_only=False)
    assert (blob["ch"], blob["n_joints"], blob["n_feat"]) == (24, 6, 99)
    model = TrajectoryEncoder(24, 16, 99).eval()
    model.load_state_dict(blob["state_dict"], strict=True)
    q = np.asarray(corpus.joint(0, "left"))
    x = ((metric_series(q) - blob["chan_mu"]) / blob["chan_sd"]).T[None].astype(np.float32)
    with torch.no_grad():
        mu, _ = model.encode(torch.tensor(x), torch.ones(1, 1, len(q)))
    np.testing.assert_allclose(encoder.encode(x)[0], mu.numpy(), rtol=1e-4, atol=1e-6)
    assert np.isfinite(prior.nll(q)).all()
    with np.load(out / "posture_ref.npz", allow_pickle=False) as z:
        assert str(z["model"]) == "gmm" and int(z["n_joints"]) == 6
        np.testing.assert_allclose(
            z["gmm_prec_chol"], np.asarray(prior.precision_factors), rtol=1e-6
        )
    before = (out / "checksums.json").read_bytes()
    train_arm(*args)
    assert before == (out / "checksums.json").read_bytes()
    (out / "checksums.json").unlink()  # Interrupted between last epoch and export.
    train_arm(*args)
    np.testing.assert_allclose(load_encoder(out / "encoder.npz").latent_mean, encoder.latent_mean)
    assert json.loads((out / "progress.json").read_text())["phase"] == "complete"
