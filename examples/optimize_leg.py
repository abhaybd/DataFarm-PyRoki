"""Refine one joint-space leg from a local URDF; saves arrays and never executes a robot."""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyroki as pk
import yourdfpy

from datafarm_pyroki import JointMap, load_encoder
from datafarm_pyroki.integration import solve_leg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--start",
        nargs="+",
        type=float,
        required=True,
        help="Actual start in PyRoki actuated joint order",
    )
    parser.add_argument(
        "--goal",
        nargs="+",
        type=float,
        required=True,
        help="Already selected endpoint in PyRoki actuated joint order",
    )
    parser.add_argument(
        "--model-joints",
        nargs="+",
        help="URDF joint names in model training order; explicit name aliases",
    )
    parser.add_argument("--seed", type=Path, help="Initial waypoint NPY in robot order")
    parser.add_argument("--dt", type=float, default=0.15)
    parser.add_argument("--timesteps", type=int, default=32)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    encoder = load_encoder(args.checkpoint)
    urdf = yourdfpy.URDF.load(args.urdf)
    robot = pk.Robot.from_urdf(urdf)
    aliases = None
    if args.model_joints is not None:
        if len(args.model_joints) != encoder.spec.n_joints:
            parser.error("--model-joints must list exactly one URDF joint per model coordinate")
        aliases = dict(zip(encoder.spec.joint_names, args.model_joints, strict=True))
    mapping = JointMap.from_names(
        robot.joints.actuated_names, encoder.spec.joint_names, aliases=aliases
    )
    seed = None if args.seed is None else np.load(args.seed, allow_pickle=False)
    result = solve_leg(
        robot,
        encoder,
        jnp.array(args.start),
        jnp.array(args.goal),
        joint_map=mapping,
        source_dt=args.dt,
        initial_waypoints=seed,
        timesteps=args.timesteps,
        encoder_weight=args.weight,
        robot_coll=pk.collision.RobotCollision.from_urdf(urdf),
        max_iterations=args.iterations,
        verbose=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        positions=np.asarray(result.positions),
        dt=result.source_dt,
        robot_joint_names=np.asarray(mapping.robot_joint_names),
        model=np.asarray(encoder.spec.name),
    )
    print(
        f"Saved {args.output}; style cost {result.initial_style_cost:.5g} -> "
        f"{result.final_style_cost:.5g}; constraint violation "
        f"{result.max_constraint_violation:.3g}"
    )


if __name__ == "__main__":
    main()
