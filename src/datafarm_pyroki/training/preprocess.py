"""Disk-backed crop cache with paired input/target windows and training-only statistics."""

from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import warnings
from pathlib import Path

import numpy as np

from .data import Corpus, atomic_json, manifest_hash
from .features import COMMON_RATE, MIN_RAW_LEN, Traj, fingerprint, metric_series

MAX_LENGTH = 768  # Same 25.6-second duration as upstream's 384 frames at 15 Hz.
_WORKER = None


def window_variants(q: np.ndarray, crops: int, seed: int, max_length: int = MAX_LENGTH):
    """Original full+6 crop policy, with matching capped positions for input AND target."""
    rng = np.random.default_rng(seed)
    for i in range(crops + 1):
        length = (
            len(q) if i == 0 else min(len(q), max(MIN_RAW_LEN, int(len(q) * rng.uniform(0.4, 1.0))))
        )
        start = int(rng.integers(0, len(q) - length + 1))
        if length > max_length:
            start += int(rng.integers(0, length - max_length + 1))
            length = max_length
        yield np.asarray(q[start : start + length], np.float64)


def _init_worker(root, arm, crops, seed, max_length):
    global _WORKER
    _WORKER = (Corpus(Path(root)), arm, crops, seed, max_length)


def _episode(index):
    corpus, arm, crops, seed, max_length = _WORKER
    count = crops if corpus.splits[index] == "train" else 0
    # Independent RNG per episode makes results invariant to worker count and scheduling.
    windows = list(window_variants(corpus.joint(index, arm), count, seed + int(index), max_length))
    targets = []
    series = []
    sums = np.zeros((2, 24), np.float64)
    total = 0
    for q in windows:
        tr = Traj(str(corpus.sources[index]), str(corpus.ids[index]), q, COMMON_RATE, len(q))
        targets.append(list(fingerprint(tr).values()))
        x = metric_series(q).astype(np.float32)
        series.append(x)
        sums[0] += x.sum(0, dtype=np.float64)
        sums[1] += np.square(x.astype(np.float64)).sum(0)
        total += len(x)
    return series, np.asarray(targets), sums, total


def fit_target_statistics(targets):
    """Stationary arms have undefined spectra; impute from training only, zero if all absent."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        impute = np.nanmedian(np.where(np.isfinite(targets), targets, np.nan), axis=0)
    impute = np.nan_to_num(impute)
    filled = np.where(np.isfinite(targets), targets, impute)
    mean, scale = filled.mean(0), filled.std(0)
    scale[scale < 1e-9] = 1.0
    return impute, mean, scale


def prepare_features(
    corpus: Corpus, arm: str, root: Path, *, crops=6, seed=0, workers=16, max_length=MAX_LENGTH
):
    recipe = {
        "version": 1,
        "manifest": corpus.digest,
        "arm": arm,
        "crops": crops,
        "seed": seed,
        "max_length": max_length,
        "sample_rate_hz": COMMON_RATE,
        "paired_window_targets": True,
    }
    digest = manifest_hash(recipe)
    root = root / arm / digest[:16]
    ready = root / "ready.json"
    if ready.exists():
        if json.loads(ready.read_text()) != recipe:
            raise ValueError("Feature-cache recipe mismatch")
        return root
    root.mkdir(parents=True, exist_ok=True)
    train_count = len(corpus.indices("train")) * (crops + 1)
    count = train_count + len(corpus.indices("val"))
    x = np.lib.format.open_memmap(
        root / "series.npy", mode="w+", dtype=np.float32, shape=(count, max_length, 24)
    )
    y = np.empty((count, 99), np.float64)
    lengths = np.empty(count, np.int32)
    sources = np.empty(count, dtype="U10")
    splits = np.empty(count, dtype="U5")
    episodes = np.empty(count, np.int64)
    order = np.concatenate((corpus.indices("train"), corpus.indices("val")))
    moments, frames, cursor = np.zeros((2, 24), np.float64), 0, 0
    print(
        f"[features] {arm}: {train_count} training windows, "
        f"{count - train_count} validation windows",
        flush=True,
    )
    initargs = (str(corpus.root), arm, crops, seed, max_length)
    if workers > 1:
        pool = cf.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=initargs,
        )
        results = pool.map(_episode, order, chunksize=8)
    else:
        pool = None
        _init_worker(*initargs)
        results = map(_episode, order)
    try:
        for i, (index, result) in enumerate(zip(order, results, strict=True), 1):
            series, targets, sums, total = result
            stop = cursor + len(series)
            for j, values in enumerate(series, cursor):
                x[j, : len(values)] = values
                lengths[j] = len(values)
            y[cursor:stop] = targets
            sources[cursor:stop] = corpus.sources[index]
            splits[cursor:stop] = corpus.splits[index]
            episodes[cursor:stop] = index
            if corpus.splits[index] == "train":
                moments += sums
                frames += total
            cursor = stop
            if i % 1000 == 0 or i == len(order):
                print(f"[features] {arm}: {i}/{len(order)} episodes", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    if cursor != count:
        raise ValueError("Incomplete crop cache")
    cmu = moments[0] / frames
    csd = np.sqrt(np.maximum(0.0, moments[1] / frames - cmu**2)) + 1e-6
    impute, fmu, fsd = fit_target_statistics(y[:train_count])
    y = ((np.where(np.isfinite(y), y, impute) - fmu) / fsd).astype(np.float32)
    names = list(fingerprint(Traj("", "", np.zeros((MIN_RAW_LEN, 6)), COMMON_RATE, MIN_RAW_LEN)))
    np.savez(
        root / "metadata.npz",
        targets=y,
        lengths=lengths,
        sources=sources,
        splits=splits,
        episodes=episodes,
        chan_mu=cmu.astype(np.float32),
        chan_sd=csd.astype(np.float32),
        feat_mu=fmu,
        feat_sd=fsd,
        feat_impute=impute,
        feat_names=np.asarray(names),
    )
    x.flush()
    del x
    atomic_json(ready, recipe)
    return root


class FeatureCache:
    def __init__(self, path):
        self.recipe = json.loads((Path(path) / "ready.json").read_text())
        # Copy-on-write mapping is writable to torch without modifying the on-disk cache.
        self.series = np.load(Path(path) / "series.npy", mmap_mode="c")
        with np.load(Path(path) / "metadata.npz", allow_pickle=False) as z:
            self.metadata = {k: z[k] for k in z.files}

    def groups(self, split):
        m = self.metadata
        return [
            np.flatnonzero((m["splits"] == split) & (m["sources"] == source))
            for source in ("abc", "molmoact2")
        ]
