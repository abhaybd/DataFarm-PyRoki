"""Prepare the balanced YAM corpus, then train both marginal models on one GPU.

Run with the training extra installed. Logs go to stdout; pipeline.json tracks
completion/failure independently of a terminal session. Re-running resumes work.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from datafarm_pyroki.training.data import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("work/yam"))
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--data-workers", type=int, default=8)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    commands = [
        (
            "prepare",
            [
                sys.executable,
                "-u",
                "-m",
                "datafarm_pyroki.training.data",
                "--root",
                str(root / "corpus"),
                "--workers",
                str(a.data_workers),
            ],
        ),
        (
            "train",
            [
                sys.executable,
                "-u",
                "-m",
                "datafarm_pyroki.training.train",
                "--corpus",
                str(root / "corpus"),
                "--out",
                str(root / "models"),
                "--feature-cache",
                str(root / "features"),
                "--gpu",
                str(a.gpu),
                "--workers",
                str(a.workers),
            ],
        ),
    ]
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {
            "pid": os.getpid(),
            "gpu": a.gpu,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        for phase, command in commands:
            status.update(phase=phase, command=command)
            atomic_json(root / "pipeline.json", status)
            print(f"[pipeline] {phase}: {' '.join(command)}", flush=True)
            try:
                subprocess.run(command, check=True)
            except BaseException as exc:
                atomic_json(
                    root / "pipeline.json",
                    dict(status, phase="failed", failed_phase=phase, error=str(exc)),
                )
                raise
        atomic_json(
            root / "pipeline.json",
            dict(status, phase="complete", finished_at=datetime.now(timezone.utc).isoformat()),
        )


if __name__ == "__main__":
    main()
