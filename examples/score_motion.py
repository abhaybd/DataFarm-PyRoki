"""Score a motion file, or a synthetic numerical smoke input, with the frozen model."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from datafarm_pyroki import load_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, help="NPY array [time,model_joints], radians")
    parser.add_argument("--checkpoint", type=Path, help="Converted NPZ; default is bundled DROID")
    parser.add_argument("--dt", type=float, default=0.15, help="Input waypoint timestep, seconds")
    args = parser.parse_args()
    encoder = load_encoder(args.checkpoint)
    if args.positions is None:
        t = np.linspace(0, 1, 32, dtype=np.float32)[:, None]
        q = 0.1 * np.sin(t * np.arange(1, encoder.spec.n_joints + 1)[None])
    else:
        q = np.load(args.positions, allow_pickle=False)
    q = jnp.asarray(q, dtype=jnp.float32)
    cost, grad = jax.jit(jax.value_and_grad(lambda x: encoder.cost(x, args.dt)))(q)
    print(
        json.dumps(
            {
                "model": encoder.spec.name,
                "joint_names": encoder.spec.joint_names,
                "input_shape": list(q.shape),
                "model_rate_hz": encoder.spec.sample_rate_hz,
                "mahalanobis_squared": float(cost),
                "waypoint_gradient_l2": float(jnp.linalg.norm(grad)),
                "finite_gradient": bool(jnp.isfinite(grad).all()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
