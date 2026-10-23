# Joint-position NLL selection

`JointPositionPrior` represents a full-covariance Gaussian mixture. It is separate
from the motion encoder and its latent reference Gaussian. The default DROID GMM
has 16 components over seven proximal-to-distal Franka arm coordinates. Its
provenance records 4,000 episodes / 987,631 non-idle frames. The source parameters
are extracted without refitting from the pinned cuTAMP `posture_ref.npz`.
Checkpoint binaries are local/downloaded artifacts; Git retains only provenance
and license metadata.

For standardized coordinates `z = (q - location) / scale`, the component density
uses `precision[k] = C[k] @ C[k].T`. Both triangular factor orientations are
supported because sklearn's stored factor is upper triangular in this convention.
The full NLL includes the Gaussian normalization and `sum(log(scale))` Jacobian.
Those terms are constant across branches of the same query; including them changes
reported NLL values but preserves DATAFARM's ranking exactly.

## Feasibility and candidate layout

`rank_ik_solutions(prior, q, joint_map=..., feasible=...)` takes candidates shaped
`[..., K, robot_dof]`, and a same-query feasibility mask `[..., K]`. The mask is
required: the caller must check pose residuals, collisions and task constraints.
The utility additionally rejects any nonfinite coordinate or NLL and can enforce
robot-order joint limits. A low NLL is never a substitute for feasibility.

Returns `IKRanking(indices, nll, feasible)`. `indices` is a stable increasing-NLL
ordering; `nll` remains in original candidate order with invalid values at infinity.
This function is JIT-compatible. Inside a JIT, inspect `feasible.any(-1)` before
using the first index; all-invalid queries have no valid choice.

`select_ik_solution` is the host-side convenience wrapper. It supports batches,
raises `NoFeasibleIKSolutions` if any query is all-invalid, and returns the complete
robot configuration, its original candidate index and its NLL. Extra robot
coordinates outside the prior mapping are preserved and still checked for finite
values/limits. Use its positions as the fixed goal of `solve_leg`.

## Other trained priors

```python
from datafarm_pyroki import JointPositionPrior

prior = JointPositionPrior.from_parameters(
    weights,
    means,
    covariances,
    joint_names=ordered_training_joint_names,
    name="my-reference-model",
    location=training_location,
    scale=training_scale,
)
prior.save("my_joint_prior.npz")
```

Covariances and means describe standardized coordinates. For a GMM fit to raw
joint values, omit location/scale. Construction validates dimensions, covariance
symmetry/positive definiteness, unique names, positive scales and mixture weights.
`load_joint_prior(path)` uses pickle-free NumPy loading. Models may have different
numbers of joints/components and coexist without shared global state.

## Artifact location and license

`load_joint_prior()` resolves the pinned artifact from
[abhaybd/DataFarm-Jax-PosturePrior-Franka](https://huggingface.co/abhaybd/DataFarm-Jax-PosturePrior-Franka),
verifies its SHA256, and reuses the standard Hugging Face cache. Use `hf auth login`,
`HF_TOKEN`, or `token=` for private-repository access. `cache_dir=` selects another
Hub cache; `local_files_only=True` requires an existing cache entry. `HF_HUB_OFFLINE=1`
provides the corresponding environment setting. An explicit NPZ path bypasses the
Hub and allows another trained model; source-tree artifacts and `DATAFARM_ASSET_DIR`
are no longer implicit defaults.

`uv run python scripts/convert_joint_prior.py` reproduces the pinned reference
from upstream. The source GMM is under the NVIDIA License (research/evaluation
use); retain its notice when redistributing through Hugging Face or another host.
User-trained parameters retain their own provenance and applicable terms.
