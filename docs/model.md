# Model and artifact contract

## Architecture

The inference port follows DATAFARM `encoder/model.py` at commit
`fd403ea6eb16163261a13439fe9bbc10a980f50d`.

- Input: standardized `[batch, 4*J, time]`, with joint position, velocity,
  acceleration and jerk channel groups in that order.
- Filterbank: six Conv1D branches, 32 channels each, with `(kernel, dilation)`
  `(3,1), (7,1), (15,1), (7,2), (15,4), (15,8)`. BatchNorm, absolute value,
  then masked mean and standard deviation: 384 features.
- Temporal branch: Conv1D widths `4*J -> 64 -> 96 -> 96`, kernels `5,5,3`,
  stride 2 each; BatchNorm and exact GELU. Masked mean/std/max: 288 features.
- Concatenated 672 features: Linear 256, LayerNorm, exact GELU, then Linear 96,
  LayerNorm, exact GELU. Dropout is disabled at inference.
- Linear output: `2*D` values split into latent mean and log variance.
- Auxiliary training head: `D -> 128 -> 113`, exact GELU between layers.

All BatchNorm layers use saved running moments with epsilon `1e-5`; LayerNorm
also uses `1e-5`. Temporal padding and endpoint-aligned interpolation match
PyTorch. JAX's default approximate GELU is deliberately not used.

`encode(..., padding_exact=False)` reproduces the original eval-mode network.
`encode(..., padding_exact=True)` zeros the temporal branch outside its validity
mask after each layer, matching the prefix-preserving runtime convention used
by the retimer. Supply zero features outside the mask in this mode. Ordinary
masked pooling alone does not eliminate bias leakage from padded convolutions.

## Motion features and cost

The source leg has a fixed timestep. Resample positions to
`max(2, round((T-1)*source_dt*sample_rate_hz)+1)` frames with endpoint-aligned
linear interpolation. Recompute derivatives using `1/sample_rate_hz`, central
interior and one-sided endpoint differences, repeated three times. This follows
the released fixed-clock cuRobo cost, including its nominal frame-count rounding.

The deterministic encoder mean is the embedding. Let `P` be the reference
precision, `P = C @ C.T` its lower Cholesky factor, and `dz = z - mean` a row vector.
The least-squares residual is `sqrt(weight) * dz @ C`, so its squared norm is
`weight * dz @ P @ dz.T`. Using `C.T` here would implement the wrong metric.
The reference Gaussian is fixed during planning. No GP is involved.

Float32 interpolation differences of a few ULP can be amplified by third
differences and may switch an absolute-value or maximum branch near zero. Tests
therefore check forward outputs and full gradient norms/directions at float32,
and also check coordinate gradients at float64. This is a numerical parity test,
not a claim of bitwise equality between frameworks.

## Default model loading

`load_encoder()` downloads `droid_encoder.npz` from
[abhaybd/DataFarm-Jax-MotionEncoder-Franka](https://huggingface.co/abhaybd/DataFarm-Jax-MotionEncoder-Franka)
at the revision pinned in `assets.py`, verifies its SHA256, and uses the standard
Hugging Face cache. Use `hf auth login`/`HF_TOKEN` or `token=` for private access,
`cache_dir=` to override the Hub cache, and `local_files_only=True` to load an
already cached model without network access. Explicit NPZ paths bypass Hub
resolution so custom trained models still work. Binary files under the source
asset folder are not consulted by default.

## NPZ schema (version 1)

`load_encoder(path)` uses `allow_pickle=False`. Required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | scalar integer, 1 |
| `model_name` | model identity string |
| `joint_names` | ordered, unique training coordinate names, length J |
| `sample_rate_hz` | positive training sample rate |
| `state_dict/<name>` | original PyTorch tensor layout, including buffers |
| `chan_mu`, `chan_sd` | input normalization vectors, length 4*J |
| `reference_mean` | reference latent mean, length D |
| `reference_precision` | positive-definite D-by-D precision |
| `reference_cov` | optional reference covariance, retained by converter |

Tensor layouts stay in PyTorch order; JAX convolutions explicitly use NCH/OIH
layouts. Model input and latent dimensions are inferred from the tensors and
checked against the metadata. Models are loaded independently; multiple model
objects can coexist without a global cache or shared reference statistics.

For another checkpoint with the same encoder architecture, use:

```sh
uv run python scripts/convert_checkpoint.py \
  --checkpoint path/to/trained_encoder.pt \
  --output path/to/model.npz \
  --sha256 YOUR_CHECKPOINT_SHA256 \
  --model-name YOUR_MODEL_ID \
  --joint-names joint_a joint_b joint_c \
  --sample-rate-hz 15
```

Supply all training joint names in order. The source checkpoint must have the
architecture's `state_dict`, `ch`, `n_joints`, normalization, and reference
statistics. Statistics may be named `reference_mean/cov/precision` or the original
`droid_latent_mean/cov/precision`. A different network architecture needs a separate
implementation; changing metadata cannot adapt trained weights to a new robot.
The restricted PyTorch loader permits the NumPy types needed by the original
artifact; unsupported serialization types fail conversion explicitly.

## Provenance discrepancy

The reference artifact records a mixed DROID/cuTAMP training protocol with a Fisher
latent-discrimination term. The current upstream training source describes
DROID-only fingerprint regression with beta-KL regularization. The exported model
retains the shipped checkpoint's metadata, rather than labeling it with the newer
script's protocol. Its reference mean/precision remain fitted to DROID embeddings.
