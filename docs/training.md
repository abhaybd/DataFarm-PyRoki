# Training left/right YAM marginals

The optional PyTorch pipeline trains four independent models: a six-dimensional
posture GMM and a six-joint motion encoder for each arm. No DATAFARM checkout,
cuTAMP, cuRobo, or tiptop installation is required. Training code lives in
`src/datafarm_pyroki/training/`; normal inference does not import it.

```sh
uv sync --extra training
uv run --extra training python -u scripts/train_yam.py --gpu 3
```

The wrapper prepares data and then trains both arms. `work/yam/pipeline.json`
records its phase or failure. To run either stage separately, use
`python -m datafarm_pyroki.training.data` or
`JAX_PLATFORMS=cpu python -m datafarm_pyroki.training.train --gpu 3` in the training
environment.

Both commands are resumable. Preparation caches completed data files and refuses
to mix different sampling manifests. Training saves optimizer, scheduler, best
weights, and NumPy/PyTorch random states after each epoch in `resume.pt`. Reissue
the same command to continue. A completed run verifies its artifact checksums and
skips training. Different settings require a new `--root` or `--out`, respectively.
The default output is `work/yam/models/{left,right}/`; all data and binaries are
ignored by Git. `--help` lists smaller smoke budgets and CPU options. Training
the arms sequentially keeps GPU memory use bounded. `--no-preload` leaves the crop
cache on the host; by default it is copied to the selected GPU.

## Sources and coordinate contract

The sources are pinned and only tabular proprioception is read. Videos and images
are not downloaded. Remote Parquet reads can include adjacent bytes for efficient
HTTP buffering; actions are never used as training observations.

| Source | Pinned revision | State coordinates |
| --- | --- | --- |
| [ABC LeRobot v3 train](https://huggingface.co/datasets/lerobot/abc_130k_v3_train) | `68651e4929d9fb00f798937b2d62617cab5c771d` | `left_arm_joint_1..6`, gripper, `right_arm_joint_1..6`, gripper |
| [MolmoAct2 Bimanual YAM](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset) | `e9f21ae15074330839f2ac25ed4b49d76dfa1f9c` | `left_joint_0..5.pos`, gripper, `right_joint_0..5.pos`, gripper |

Both publish observed positions on a **30 Hz recording clock**. This is the
dataset sampling frequency, not a claim about the motor servo-loop frequency.
We preserve these samples without downsampling or time normalization. Timestamps
must be strictly increasing with approximately 1/30-second spacing (2 ms absolute
tolerance for float32 timestamps); missing frames and large gaps are rejected.
Episodes spanning multiple Parquet files are assembled before validation.

The canonical model names are `left_joint1..6` and `right_joint1..6`, proximal to
distal, in **radians**, with both grippers excluded. Extraction uses explicit
metadata names, not just matching dimensions. There is no left/right mirroring,
degree conversion, angle wrapping, or per-source centering.

The convention audit compared the published robot paths:

- [ABC's I2RT dependency](https://github.com/arthurallshire/i2rt-drivers/blob/852f6fff33970fe9b882c53e6b28c66172c97291/i2rt/robots/get_robot.py)
- [MolmoAct2's pinned YAM SDK](https://github.com/Everloom-129/YAM/blob/9f06bba2a36dd84fb36d0c31337c85d4bf1cea22/i2rt/i2rt/robots/get_robot.py)
- [MolmoAct2's state conversion](https://github.com/allenai/molmoact2/blob/main/examples/yam/lerobot_convert.py)

Both SDKs use the same six arm motor IDs, zero nominal offsets, positive motor
directions, and arm limits. Their startup calibration adjusts equivalent full
turns; the published corpus ranges also agree with these native radians. The
pipeline rejects extreme coordinates and single-frame jumps over pi rather than
guessing a corrective transform. This establishes the published convention; it
does not independently certify every physical station's calibration.

## Sampling and training

Each source contributes **15,000 unique training episodes and 600 validation
episodes**, sampled uniformly without replacement with fixed seeds. Episodes
shorter than 60 frames (about two seconds) are ineligible. The two arms use exactly
the same episode splits. No task balancing or deduplication is performed.
The selected IDs, revisions, extraction mapping, and clock are saved in
`work/yam/corpus/manifest.json`. `corpus.json` records the clock audit and manifest
hash; every model report carries that hash.

Defaults follow the current upstream DATAFARM training script:

- Original filterbank/temporal encoder; 24 input channels `[q | v | a | jerk]`,
  16-dimensional Gaussian latent, 99 auxiliary fingerprint targets.
- Each episode plus six seeded crops spanning 40–100% of the episode. Each
  minibatch contains 64 uniformly sampled crop entries from each source.
- 60 epochs, 800 steps/epoch, batch 128, Adam at 7e-4 with cosine annealing,
  weight decay 1e-4, gradient clipping at 5, dropout 0.3, input noise 0.1.
- Fingerprint regression with band/spectral weights 4 and beta-KL with free bits
  0.1/dimension; beta 0.005 after a 10-epoch linear warmup.
- Select the lowest held-out validation loss, evaluated every third epoch and
  at the last epoch. Reports include losses separately for each source.

The YAM clock is **30 Hz throughout**. Derivatives use 1/30-second differences.
The maximum training window is 768 frames (25.6 seconds); latent reference
segments are 100–190 frames (about 3.3–6.3 seconds). These preserve upstream's
physical window durations at twice its 15 Hz sampling frequency. The high
spectral band extends to the YAM Nyquist frequency of 15 Hz. Convolution kernels
and dilations retain the original architecture; their time spans in seconds are
therefore half the Franka model's.

There are deliberate corrections to upstream's data handling:

- The fingerprint target and derivative inputs are computed from the **same
  capped position window**. Upstream can cap an input while retaining the longer
  crop's fingerprint target.
- Validation uses one deterministic capped window per episode. Its episode IDs
  remain disjoint from all training, normalization, and reference fitting.
- Fingerprint imputation and all normalization are fitted on training data only.
  An entirely undefined descriptor (e.g. spectra of an always stationary joint)
  gets zero rather than propagating NaN.
- Latent reference statistics use **120,000 training-only segments**, 60,000 per
  source, with approximately equal weight per episode. Upstream calibrates on all
  cached episodes, including validation. A 1e-4 covariance ridge is retained.

Each posture GMM uses 16 full-covariance components. Per arm, select 2,000 training
episodes uniformly per source, retain frames with joint-speed norm above
0.05 rad/s, and sample **60,000 frames per source** without replacement. Fit a
single standardization on that balanced pool and use covariance regularization
1e-4. Held-out non-idle frames report NLL separately for each source. This keeps
upstream's 4,000-episode/120,000-frame scale while replacing its file-order episode
selection with uniform sampling. Fitting uses scikit-learn; runtime parameters
are framework-independent and match the original PyTorch scorer's layout.

## Artifacts

An opt-in [bimanual experiment](bimanual_encoder.md) trains one synchronized
12-joint motion encoder on the identical episodes and crop windows while
retaining the existing separate posture priors:

```sh
uv run --extra training python scripts/train_yam.py --bimanual --gpu 3
```

It writes only encoder artifacts under `work/yam/models/bimanual/` (48 input
channels, 183 fingerprint targets, 16 latent dimensions). Marginal artifacts
and the default left/right training command are unchanged.

Each arm's directory contains:

| File | Consumer |
| --- | --- |
| `encoder.pt` | Original PyTorch architecture and DATAFARM checkpoint fields |
| `posture_ref.npz` | Original posture GMM field names and sklearn precision factors |
| `encoder.npz` | This repository's pickle-free JAX encoder loader |
| `posture_prior.npz` | This repository's pickle-free JAX posture loader |
| `run.json`, `encoder_report.json`, `posture_report.json` | Configuration, sampling provenance, metrics |
| `checksums.json`, `progress.json` | Final artifact hashes and current phase |

The encoder preserves historical `droid_latent_mean/cov/precision` keys, containing
**YAM** reference statistics. Both `source_rate` and `sample_rate_hz` are 30;
`n_joints=6`, `ch=24`, `n_feat=99`. Joint names are explicit. Original consumers
must honor this rate and joint mapping; planner integration is outside this
pipeline's scope. JAX exports use the exact same learned tensors and statistics.

```python
from datafarm_pyroki import load_encoder, load_joint_prior

left_encoder = load_encoder("work/yam/models/left/encoder.npz")
left_posture = load_joint_prior("work/yam/models/left/posture_prior.npz")
# q_left: [time,6], canonical left arm radians, sampled at 30 Hz
latent = left_encoder.latent(q_left, source_dt=1 / 30)
posture_nll = left_posture.nll(q_left)
```

`model.py` and the feature algorithms are vendored from MIT-licensed
[DATAFARM fd403ea](https://github.com/SamratSahoo/DATAFARM/tree/fd403ea6eb16163261a13439fe9bbc10a980f50d/encoder),
with the upstream notice included beside the modules. The trainer retains its
loss/optimization recipe with the documented data-handling changes. The shipped
Franka checkpoint records a different older mixed-data training protocol; this
pipeline follows the available training source, not a claimed exact reproduction
of that checkpoint. Posture fitting uses a local implementation of standard GMM
fitting; it does not use the pretrained Franka prior.
