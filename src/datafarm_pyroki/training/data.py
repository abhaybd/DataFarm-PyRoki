"""Prepare a reproducible, balanced, proprioception-only YAM corpus from LeRobot v3.

Run ``python -m datafarm_pyroki.training.data --help``. No images/actions are fetched.
Observed arm positions retain the native I2RT YAM radians, signs, and zeros.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

RATE = 30.0
MIN_LENGTH = 60
SOURCES = {
    "abc": {
        "repo": "lerobot/abc_130k_v3_train",
        "revision": "68651e4929d9fb00f798937b2d62617cab5c771d",
        "state_names": [
            name
            for arm in ("left", "right")
            for name in (*[f"{arm}_arm_joint_{i}" for i in range(1, 7)], f"{arm}_gripper")
        ],
    },
    "molmoact2": {
        "repo": "allenai/MolmoAct2-BimanualYAM-Dataset",
        "revision": "e9f21ae15074330839f2ac25ed4b49d76dfa1f9c",
        "state_names": [
            name
            for arm in ("left", "right")
            for name in (*[f"{arm}_joint_{i}.pos" for i in range(6)], f"{arm}_gripper.pos")
        ],
    },
}
JOINT_NAMES = tuple(f"{arm}_joint{i}" for arm in ("left", "right") for i in range(1, 7))
ARM_COLUMNS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def manifest_hash(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def state_columns(info: dict, source: str) -> tuple[int, ...]:
    """Fail closed on unfamiliar layouts; select named arms without either gripper."""
    if float(info["fps"]) != RATE:
        raise ValueError(f"{source}: expected native {RATE} Hz, got {info['fps']}")
    feature = info["features"]["observation.state"]
    names = feature["names"]
    expected = SOURCES[source]["state_names"]
    if feature["shape"] != [14] or len(names) != 14 or set(names) != set(expected):
        raise ValueError(f"{source}: unsupported observation.state layout: {feature}")
    return tuple(names.index(expected[i]) for i in ARM_COLUMNS)


def choose_splits(rows: list[dict], train: int, val: int, seed: int) -> dict[str, list[int]]:
    """Uniform episode sampling without replacement, before extracting either arm."""
    ids = sorted(int(r["episode_index"]) for r in rows if int(r["length"]) >= MIN_LENGTH)
    if len(ids) != len(set(ids)):
        raise ValueError("Episode metadata contains duplicate episode IDs")
    if min(train, val) < 1 or len(ids) < train + val:
        raise ValueError(f"Need {train + val} eligible unique episodes, found {len(ids)}")
    selected = np.random.default_rng(seed).permutation(ids)
    return {
        "train": sorted(selected[:train].tolist()),
        "val": sorted(selected[train : train + val].tolist()),
    }


def canonical_episode(q, timestamps, frames, expected_length: int) -> tuple[np.ndarray, dict]:
    """Assemble file-spanning episodes and validate their real 30 Hz recorded clock.

    No angle wrapping, mirroring, source-specific standardization, or resampling.
    Unexpected discontinuities are errors, never silently smoothed away.
    """
    order = np.argsort(frames, kind="stable")
    q, t, f = (
        np.asarray(q)[order],
        np.asarray(timestamps, np.float64)[order],
        np.asarray(frames)[order],
    )
    if q.shape != (expected_length, 12) or not np.array_equal(f, np.arange(expected_length)):
        raise ValueError(
            f"Missing/duplicate frames: expected ({expected_length},12), got {q.shape}"
        )
    if not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError("Nonfinite joint positions or timestamps")
    dt = np.diff(t)
    if len(q) < MIN_LENGTH or not np.allclose(dt, 1 / RATE, rtol=0, atol=2e-3):
        raise ValueError(f"Invalid native clock: dt range {dt.min()}..{dt.max()}")
    # The published I2RT arm limits are within about +/-4 radians. Wide bounds catch
    # degrees, accidental gripper/effort columns, and corrupted multi-turn coordinates.
    if np.max(np.abs(q)) > 4.5:
        raise ValueError("Joint position outside supported YAM radian range (+/-4.5)")
    if np.max(np.abs(np.diff(q, axis=0))) > np.pi:
        raise ValueError("Angular discontinuity exceeds pi in one recorded frame")
    return q.astype(np.float32), {"dt_min": float(dt.min()), "dt_max": float(dt.max())}


def _metadata(root: Path, source: str):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    spec = SOURCES[source]
    cache = root / "hub"
    files = HfApi().list_repo_files(spec["repo"], repo_type="dataset", revision=spec["revision"])

    def get(path):
        return hf_hub_download(
            spec["repo"], path, repo_type="dataset", revision=spec["revision"], cache_dir=cache
        )

    info = json.loads(Path(get("meta/info.json")).read_text())
    cols = state_columns(info, source)
    rows = []
    columns = [
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
    ]
    for path in sorted(
        p for p in files if p.startswith("meta/episodes/") and p.endswith(".parquet")
    ):
        rows.extend(pq.read_table(get(path), columns=columns).to_pylist())
    return (
        info,
        cols,
        rows,
        sorted(p for p in files if p.startswith("data/") and p.endswith(".parquet")),
    )


def _fetch_file(root: Path, source: str, path: str, selected: set[int], cols: tuple[int, ...]):
    """Project selected row groups/columns; retain fragments until episode assembly."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    spec = SOURCES[source]
    out = root / "fragments" / source / (path.replace("/", "_") + ".npz")
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    remote = f"datasets/{spec['repo']}@{spec['revision']}/{path}"
    selected_sorted = np.sort(list(selected))
    for attempt in range(7):
        try:
            fs = HfFileSystem()
            # Large ranges amortize CDN redirects for episode-sized Parquet row groups.
            # These files contain tabular data only; camera media lives elsewhere.
            with fs.open(remote, "rb", block_size=64 * 1024 * 1024) as f:
                pf = pq.ParquetFile(f)
                epcol = pf.schema_arrow.names.index("episode_index")
                groups = []
                for i in range(pf.num_row_groups):
                    stats = pf.metadata.row_group(i).column(epcol).statistics
                    intersects = stats is None or not stats.has_min_max
                    if not intersects:
                        lo = np.searchsorted(selected_sorted, stats.min, side="left")
                        hi = np.searchsorted(selected_sorted, stats.max, side="right")
                        intersects = hi > lo
                    if intersects:
                        groups.append(i)
                table = pf.read_row_groups(
                    groups,
                    columns=["observation.state", "timestamp", "frame_index", "episode_index"],
                )
            ids = table["episode_index"].to_numpy()
            mask = np.isin(ids, list(selected))
            table = table.filter(mask)
            ids = ids[mask]
            state = np.asarray(table["observation.state"].to_pylist(), np.float32).reshape(-1, 14)
            ts, frames = table["timestamp"].to_numpy(), table["frame_index"].to_numpy()
            arrays = {}
            for ep in np.unique(ids):
                m = ids == ep
                arrays[f"q_{ep}"] = state[m][:, cols]
                arrays[f"t_{ep}"] = ts[m]
                arrays[f"f_{ep}"] = frames[m]
            tmp = out.with_suffix(".tmp.npz")
            np.savez(tmp, **arrays)
            tmp.replace(out)
            return out
        except Exception as exc:
            if attempt == 6:
                raise
            delay = min(60, 2 ** (attempt + 1))
            print(
                f"[fetch] retry {source}/{path}: {type(exc).__name__}: {str(exc)[:180]} ({delay}s)",
                flush=True,
            )
            time.sleep(delay)


def prepare(root: Path, train: int, val: int, seed: int, workers: int):
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "sample_rate_hz": RATE,
        "joint_names": JOINT_NAMES,
        "units": "radians",
        "convention": "I2RT YAM native arm coordinates; identity signs/offsets",
        "train_per_source": train,
        "val_per_source": val,
        "seed": seed,
        "sources": {},
    }
    metadata = {}
    for k, source in enumerate(SOURCES):
        info, cols, rows, paths = _metadata(root, source)
        splits = choose_splits(rows, train, val, seed + k)
        manifest["sources"][source] = dict(
            SOURCES[source],
            splits=splits,
            state_columns=cols,
            total_episodes=info["total_episodes"],
        )
        metadata[source] = rows, paths
    # Persist before network work; resumed runs cannot mix selections or revisions.
    manifest = json.loads(json.dumps(manifest))
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing corpus manifest differs; use a new --root")
    atomic_json(path, manifest)
    if (root / "corpus.json").exists():
        if json.loads((root / "corpus.json").read_text())["manifest_sha256"] != manifest_hash(
            manifest
        ):
            raise ValueError("Corpus/manifest mismatch")
        print("[data] prepared corpus already complete", flush=True)
        return
    all_rows, all_sources, all_ids, all_splits, chunks = [], [], [], [], []
    audits = {}
    for source, spec in manifest["sources"].items():
        rows, paths = metadata[source]
        selected = set(spec["splits"]["train"] + spec["splits"]["val"])
        # An episode may span files. Include its start through the NEXT episode's
        # start file, conservatively inclusive, then assemble by frame_index below.
        ordered = sorted(rows, key=lambda r: r["dataset_from_index"])
        needed = set()
        for i, row in enumerate(ordered):
            if row["episode_index"] not in selected:
                continue
            start = (
                f"data/chunk-{row['data/chunk_index']:03d}/"
                f"file-{row['data/file_index']:03d}.parquet"
            )
            stop = paths[-1]
            if i + 1 < len(ordered):
                nxt = ordered[i + 1]
                stop = (
                    f"data/chunk-{nxt['data/chunk_index']:03d}/"
                    f"file-{nxt['data/file_index']:03d}.parquet"
                )
            a, b = paths.index(start), paths.index(stop)
            needed.update(paths[a : b + 1])
        print(
            f"[data] {source}: {len(selected)} episodes from {len(needed)} data files (no media)",
            flush=True,
        )
        fragments = []
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            tasks = [
                pool.submit(_fetch_file, root, source, p, selected, tuple(spec["state_columns"]))
                for p in sorted(needed)
            ]
            for i, future in enumerate(cf.as_completed(tasks), 1):
                fragments.append(future.result())
                if i % 25 == 0 or i == len(tasks):
                    print(f"[data] {source}: {i}/{len(tasks)} files cached", flush=True)
        pieces = {ep: [] for ep in selected}
        for fragment in sorted(fragments):
            with np.load(fragment, allow_pickle=False) as z:
                for key in z.files:
                    if key.startswith("q_"):
                        ep = int(key[2:])
                        pieces[ep].append((z[key], z[f"t_{ep}"], z[f"f_{ep}"]))
        lengths = {int(r["episode_index"]): int(r["length"]) for r in rows}
        val_ids = set(spec["splits"]["val"])
        audit = []
        for ep in sorted(selected):
            if not pieces[ep]:
                raise ValueError(f"{source}/{ep}: selected episode missing from data")
            q, ts, frames = (np.concatenate(x) for x in zip(*pieces[ep]))
            try:
                q, report = canonical_episode(q, ts, frames, lengths[ep])
            except ValueError as exc:
                raise ValueError(f"{source}/{ep}: {exc}") from exc
            chunks.append(q)
            all_rows.append(len(q))
            all_sources.append(source)
            all_ids.append(ep)
            all_splits.append("val" if ep in val_ids else "train")
            audit.append(report)
        audits[source] = {
            "episodes": len(selected),
            "frames": sum(lengths[e] for e in selected),
            "dt_min": min(a["dt_min"] for a in audit),
            "dt_max": max(a["dt_max"] for a in audit),
        }
    positions_tmp = root / "positions.tmp.npy"
    data = np.lib.format.open_memmap(
        positions_tmp, mode="w+", dtype=np.float32, shape=(sum(all_rows), 12)
    )
    offsets = np.concatenate(([0], np.cumsum(all_rows)))
    for i, q in enumerate(chunks):
        data[offsets[i] : offsets[i + 1]] = q
    data.flush()
    del data
    positions_tmp.replace(root / "positions.npy")
    np.savez(
        root / "index.npz",
        offsets=offsets,
        sources=np.asarray(all_sources),
        episode_ids=np.asarray(all_ids),
        splits=np.asarray(all_splits),
    )
    atomic_json(
        root / "corpus.json",
        {
            "manifest_sha256": manifest_hash(manifest),
            "audit": audits,
            "positions_bytes": os.path.getsize(root / "positions.npy"),
        },
    )
    print(f"[data] complete: {audits}", flush=True)


class Corpus:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.digest = manifest_hash(self.manifest)
        if json.loads((self.root / "corpus.json").read_text())["manifest_sha256"] != self.digest:
            raise ValueError("Corpus manifest checksum mismatch")
        self.positions = np.load(self.root / "positions.npy", mmap_mode="r")
        with np.load(self.root / "index.npz", allow_pickle=False) as z:
            self.offsets, self.sources, self.ids, self.splits = (
                z[k] for k in ("offsets", "sources", "episode_ids", "splits")
            )

    def indices(self, split: str, source: str | None = None):
        mask = self.splits == split
        if source is not None:
            mask &= self.sources == source
        return np.flatnonzero(mask)

    def joint(self, index: int, arm: str) -> np.ndarray:
        columns = {"left": slice(0, 6), "right": slice(6, 12), "bimanual": slice(0, 12)}[arm]
        return self.positions[self.offsets[index] : self.offsets[index + 1], columns]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("work/yam/corpus"))
    p.add_argument("--train-per-source", type=int, default=15000)
    p.add_argument("--val-per-source", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    prepare(a.root, a.train_per_source, a.val_per_source, a.seed, a.workers)


if __name__ == "__main__":
    main()
