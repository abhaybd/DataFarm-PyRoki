"""Train separate left/right YAM posture priors and original-architecture PyTorch encoders.

Run ``python -m datafarm_pyroki.training.train --help``. Defaults match DATAFARM's
optimizer, loss, latent size, episode/crop counts, and training budget. YAM uses 30 Hz.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .artifacts import UPSTREAM_COMMIT, export_encoder, save_torch, write_checksums
from .data import JOINT_NAMES, Corpus, atomic_json
from .features import COMMON_RATE, feature_weights, metric_series, pad_mask
from .model import TrajectoryEncoder, beta_at, kl_freebits
from .posture import fit_posture
from .preprocess import FeatureCache, prepare_features


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    steps: int = 800
    batch: int = 128
    crops: int = 6
    latent: int = 16
    beta: float = 0.005
    warmup: int = 10
    free_bits: float = 0.1
    dropout: float = 0.3
    lr: float = 7e-4
    band_weight: float = 4.0
    noise: float = 0.1
    seed: int = 0
    stats_segments: int = 120000
    stats_batch: int = 256
    eval_batch: int = 256

    def __post_init__(self):
        for field in (
            "epochs",
            "steps",
            "batch",
            "latent",
            "stats_segments",
            "stats_batch",
            "eval_batch",
        ):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if self.batch % 2 or self.stats_segments % 2:
            raise ValueError("Batch and reference-segment counts must be even for 50/50 sampling")
        if self.crops < 0:
            raise ValueError("crops must be nonnegative")


def objective(pred, target, mu, logvar, weights, beta, free_bits):
    return (weights * (pred - target).square()).mean() + beta * kl_freebits(mu, logvar, free_bits)


class Batches:
    def __init__(self, cache: FeatureCache, device, preload):
        self.device = torch.device(device)
        self.metadata = cache.metadata
        self.series = torch.from_numpy(cache.series)
        if preload:
            print(
                f"[encoder] loading {cache.series.nbytes / 2**30:.2f} GiB crop cache onto {device}",
                flush=True,
            )
            self.series = self.series.to(device)
        self.lengths = torch.as_tensor(cache.metadata["lengths"], device=self.series.device)
        self.mu = torch.as_tensor(cache.metadata["chan_mu"], device=device)
        self.sd = torch.as_tensor(cache.metadata["chan_sd"], device=device)
        self.targets = torch.as_tensor(cache.metadata["targets"], device=device)

    def get(self, indices):
        lengths = self.metadata["lengths"][indices]
        size = int(lengths.max())
        ids = torch.as_tensor(indices, device=self.series.device)
        x = self.series[ids, :size].to(self.device)
        m = (
            torch.arange(size, device=self.device)[None, :]
            < torch.as_tensor(lengths, device=self.device)[:, None]
        ).float()
        x = ((x - self.mu) / self.sd) * m[:, :, None]
        return x.transpose(1, 2).contiguous(), m[:, None], self.targets[indices]


def validation_loss(model, batches, ids, weights, cfg):
    model.eval()
    gen = torch.Generator(device=batches.device).manual_seed(cfg.seed)
    mus, logvars, predictions, targets = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(ids), cfg.eval_batch):
            x, mask, y = batches.get(ids[start : start + cfg.eval_batch])
            mu, lv = model.encode(x, mask)
            eps = torch.randn(mu.shape, generator=gen, device=mu.device)
            predictions.append(model.aux(mu + eps * (0.5 * lv).exp()))
            mus.append(mu)
            logvars.append(lv)
            targets.append(y)
    return float(
        objective(
            torch.cat(predictions),
            torch.cat(targets),
            torch.cat(mus),
            torch.cat(logvars),
            weights,
            cfg.beta,
            cfg.free_bits,
        )
    )


def train_encoder(cache: FeatureCache, cfg: TrainConfig, device, out: Path, *, preload=True):
    torch.manual_seed(cfg.seed)
    model = TrajectoryEncoder(24, cfg.latent, 99, p=cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
    weights = torch.as_tensor(
        feature_weights(cache.metadata["feat_names"].tolist(), cfg.band_weight), device=device
    )
    groups = cache.groups("train")
    val_groups = cache.groups("val")
    if any(len(g) == 0 for g in groups + val_groups):
        raise ValueError("Both train and validation require both sources")
    ids = np.concatenate(val_groups)
    rng = np.random.default_rng(cfg.seed)
    batches = Batches(cache, device, preload)
    best, best_epoch, best_state = float("inf"), -1, None
    history, first_epoch = [], 0
    resume_path = out / "resume.pt"
    identity = {"config": asdict(cfg), "feature_recipe": cache.recipe}
    if resume_path.exists():
        # This is our own local optimizer checkpoint, not a downloaded model artifact.
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume["identity"] != identity:
            raise ValueError("Resume configuration/corpus differs; choose another --out")
        model.load_state_dict(resume["state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        rng.bit_generator.state = resume["numpy_rng"]
        torch.set_rng_state(resume["torch_rng"])
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state(resume["cuda_rng"], device=device)
        best, best_epoch, best_state = resume["best"], resume["best_epoch"], resume["best_state"]
        first_epoch, history = resume["epoch"] + 1, resume["history"]
        print(f"[encoder] resuming at epoch {first_epoch + 1}", flush=True)
    for epoch in range(first_epoch, cfg.epochs):
        started = time.monotonic()
        model.train()
        total = torch.zeros((), device=device)
        beta = beta_at(epoch, cfg.beta, cfg.warmup)
        for step in range(cfg.steps):
            # Exactly half the minibatch from each source. Uniform within source's crops.
            selected = np.concatenate(
                [rng.choice(g, cfg.batch // 2, replace=len(g) < cfg.batch // 2) for g in groups]
            )
            rng.shuffle(selected)
            x, mask, target = batches.get(selected)
            x = x + cfg.noise * torch.randn_like(x)
            mu, lv, pred = model(x, mask)
            loss = objective(pred, target, mu, lv, weights, beta, cfg.free_bits)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, step {step + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach()
            if (step + 1) % 200 == 0:
                print(
                    f"[encoder] epoch {epoch + 1}/{cfg.epochs} step {step + 1}/{cfg.steps} "
                    f"loss={float(total) / (step + 1):.5f}",
                    flush=True,
                )
        scheduler.step()
        record = {"epoch": epoch + 1, "train_loss": float(total) / cfg.steps, "beta": beta}
        if epoch % 3 == 2 or epoch == cfg.epochs - 1:
            val = validation_loss(model, batches, ids, weights, cfg)
            if not np.isfinite(val):
                raise FloatingPointError("Nonfinite validation loss")
            record["val_loss"] = val
            record["val_by_source"] = {
                source: validation_loss(model, batches, group, weights, cfg)
                for source, group in zip(("abc", "molmoact2"), val_groups, strict=True)
            }
            if val < best:
                best, best_epoch = val, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        record["seconds"] = time.monotonic() - started
        history.append(record)
        print(f"[encoder] {record} best_epoch={best_epoch} best_val={best:.6f}", flush=True)
        resume = dict(
            identity=identity,
            epoch=epoch,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            numpy_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None,
            best=best,
            best_epoch=best_epoch,
            best_state=best_state,
            history=history,
        )
        save_torch(resume_path, resume)
        atomic_json(
            out / "progress.json",
            {
                "phase": "encoder",
                "epoch": epoch + 1,
                "epochs": cfg.epochs,
                "latest": record,
                "best_epoch": best_epoch,
            },
        )
    if best_state is None:
        raise RuntimeError("No finite model selected")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    del batches
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return model, {"selected_epoch": best_epoch, "val_loss": best, "history": history}


def reference_statistics(model, corpus: Corpus, arm: str, metadata, cfg, device):
    """Exactly 50/50 source segments, equal episode weight, training episodes only."""
    rng = np.random.default_rng(cfg.seed + 100)
    means, kls, labels = [], [], []
    model.eval()
    for source in ("abc", "molmoact2"):
        ids = corpus.indices("train", source)
        order = np.resize(rng.permutation(ids), cfg.stats_segments // 2)
        for start in range(0, len(order), cfg.stats_batch):
            series = []
            for i in order[start : start + cfg.stats_batch]:
                q = corpus.joint(i, arm)
                length = min(len(q), int(rng.integers(100, 191)))
                offset = int(rng.integers(0, len(q) - length + 1))
                series.append(
                    metric_series(np.asarray(q[offset : offset + length], np.float64)).astype(
                        np.float32
                    )
                )
            x, mask = pad_mask(series, metadata["chan_mu"], metadata["chan_sd"])
            with torch.no_grad():
                mu, lv = model.encode(x.to(device), mask.to(device))
                means.append(mu.cpu().numpy())
                kls.append((0.5 * (mu.square() + lv.exp() - 1 - lv).sum(-1)).cpu().numpy())
                labels.extend([source] * len(mu))
            if (start // cfg.stats_batch) % 50 == 0:
                print(
                    f"[stats] {arm}/{source}: "
                    f"{min(start + cfg.stats_batch, len(order))}/{len(order)} segments",
                    flush=True,
                )
    z, kl = np.concatenate(means), np.concatenate(kls)
    mean = z.mean(0)
    covariance = np.cov(z, rowvar=False) + 1e-4 * np.eye(cfg.latent)
    precision = np.linalg.inv(covariance)
    error = z - mean
    maha = np.einsum("ni,ij,nj->n", error, precision, error)
    stats = {
        "droid_latent_mean": mean.astype(np.float32),
        "droid_latent_cov": covariance.astype(np.float32),
        "droid_latent_precision": precision.astype(np.float32),
        "kl_droid_mean": float(kl.mean()),
        "maha2_droid_mean": float(maha.mean()),
    }
    report = {
        "segments": len(z),
        "segment_frames": [100, 190],
        "training_episodes_only": True,
        "by_source": {
            source: {
                "segments": int((np.asarray(labels) == source).sum()),
                "maha2_mean": float(maha[np.asarray(labels) == source].mean()),
            }
            for source in ("abc", "molmoact2")
        },
    }
    return stats, report


def train_arm(
    corpus,
    arm,
    cfg,
    device,
    out,
    feature_root,
    workers,
    preload,
    gmm_components,
    gmm_frames,
    gmm_episodes,
    gmm_val_frames,
):
    out.mkdir(parents=True, exist_ok=True)
    run = {
        "arm": arm,
        "manifest_sha256": corpus.digest,
        "config": asdict(cfg),
        "upstream_commit": UPSTREAM_COMMIT,
        "gmm_components": gmm_components,
        "gmm_frames": gmm_frames,
        "gmm_episodes_per_source": gmm_episodes,
        "gmm_validation_frames": gmm_val_frames,
    }
    run_path = out / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != run:
        raise ValueError("Output directory belongs to a different run")
    atomic_json(run_path, run)
    if (out / "checksums.json").exists():
        import hashlib

        for name, digest in json.loads((out / "checksums.json").read_text()).items():
            if hashlib.sha256((out / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Completed artifact checksum mismatch: {out / name}")
        print(f"[train] {arm}: verified completed run", flush=True)
        return
    posture = fit_posture(
        corpus,
        arm,
        out,
        components=gmm_components,
        frames=gmm_frames,
        episodes_per_source=gmm_episodes,
        seed=cfg.seed,
        validation_frames=gmm_val_frames,
    )
    cache = FeatureCache(
        prepare_features(corpus, arm, feature_root, crops=cfg.crops, seed=cfg.seed, workers=workers)
    )
    model, metrics = train_encoder(cache, cfg, device, out, preload=preload)
    stats, stats_report = reference_statistics(model, corpus, arm, cache.metadata, cfg, device)
    names = JOINT_NAMES[:6] if arm == "left" else JOINT_NAMES[6:]
    blob = {
        "kind": "filterbank",
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "ch": 24,
        "latent": cfg.latent,
        "n_feat": 99,
        "n_joints": 6,
        "embed_dim": cfg.latent,
        "source_rate": COMMON_RATE,
        "sample_rate_hz": COMMON_RATE,
        "joint_names": list(names),
        "model_name": f"yam-{arm}-abc-molmoact2",
        "units": "radians",
        "feat_names": cache.metadata["feat_names"].tolist(),
        **{
            k: cache.metadata[k]
            for k in ("chan_mu", "chan_sd", "feat_mu", "feat_sd", "feat_impute")
        },
        **stats,
        "hparams": asdict(cfg),
        "metrics": metrics,
        "dataset": {source: dict(spec) for source, spec in corpus.manifest["sources"].items()},
        "manifest_sha256": corpus.digest,
        "upstream_commit": UPSTREAM_COMMIT,
        "train_protocol": (
            "50/50 ABC/MolmoAct2 episodes; paired 30Hz windows; original fingerprint "
            "regression + beta-KL; train-only reference statistics"
        ),
        "n_train": len(corpus.indices("train")),
        "n_val": len(corpus.indices("val")),
    }
    save_torch(out / "encoder.pt", blob)
    export_encoder(blob, out / "encoder.npz")
    atomic_json(
        out / "encoder_report.json",
        {**run, "metrics": metrics, "reference": stats_report, "posture": posture},
    )
    write_checksums(out)
    atomic_json(
        out / "progress.json", {"phase": "complete", "selected_epoch": metrics["selected_epoch"]}
    )
    print(f"[train] {arm}: exported original PyTorch/NPZ and JAX artifacts to {out}", flush=True)
    del model, cache
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("work/yam/corpus"))
    parser.add_argument("--out", type=Path, default=Path("work/yam/models"))
    parser.add_argument("--feature-cache", type=Path, default=Path("work/yam/features"))
    parser.add_argument("--arms", nargs="+", choices=("left", "right"), default=["left", "right"])
    parser.add_argument("--gpu", type=int, default=3, help="Physical GPU index; exposed as cuda:0")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-preload", action="store_true", help="Keep crop cache in host memory")
    for name, value in asdict(TrainConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    parser.add_argument("--gmm-components", type=int, default=16)
    parser.add_argument("--gmm-frames", type=int, default=120000)
    parser.add_argument(
        "--gmm-episodes", type=int, default=2000, help="Uniform training episodes per source"
    )
    parser.add_argument("--gmm-val-frames", type=int, default=12000)
    args = parser.parse_args()
    cfg = TrainConfig(**{k: getattr(args, k) for k in asdict(TrainConfig())})
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable on requested physical GPU {args.gpu}")
        print(
            f"[device] physical GPU {args.gpu}: {torch.cuda.get_device_name(0)} "
            f"| torch {torch.__version__}",
            flush=True,
        )
    torch.set_num_threads(args.threads)
    corpus = Corpus(args.corpus)
    if corpus.manifest["sample_rate_hz"] != COMMON_RATE:
        raise ValueError("Corpus/model sample rate mismatch")
    for arm in args.arms:
        train_arm(
            corpus,
            arm,
            cfg,
            args.device,
            args.out / arm,
            args.feature_cache,
            args.workers,
            not args.no_preload,
            args.gmm_components,
            args.gmm_frames,
            args.gmm_episodes,
            args.gmm_val_frames,
        )


if __name__ == "__main__":
    main()
