"""Retime a saved trajectory segment NPZ; writes data only, never controls a robot."""

import argparse
from pathlib import Path

import jax
import numpy as np

from datafarm_pyroki import JointMap, RetimeConfig, load_encoder, retime_trajectory_segment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ with positions, dt, robot_joint_names (e.g. optimize_leg.py output)",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--model-joints", nargs="+", help="Robot names in encoder training order")
    p.add_argument("--velocity-limits", type=float, nargs="+", required=True)
    p.add_argument("--acceleration-limits", type=float, nargs="+", required=True)
    p.add_argument("--output-dt", type=float, default=0.02)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--starts", type=int, default=16)
    p.add_argument("--lead-speed", type=float, default=0.0)
    p.add_argument("--trail-speed", type=float, default=0.0)
    p.add_argument("--seed", type=int, help="Sample a latent target; default uses reference mean")
    args = p.parse_args()
    encoder = load_encoder(args.checkpoint)
    with np.load(args.input, allow_pickle=False) as z:
        positions = z["positions"]
        names = tuple(z["robot_joint_names"].tolist())
        if "times" in z:
            times = z["times"]
        else:
            times = np.arange(len(positions)) * float(z["dt"])
    aliases = None
    if args.model_joints is not None:
        if len(args.model_joints) != encoder.spec.n_joints:
            p.error("--model-joints must contain one robot name per model coordinate")
        aliases = dict(zip(encoder.spec.joint_names, args.model_joints, strict=True))
    mapping = JointMap.from_names(names, encoder.spec.joint_names, aliases=aliases)
    result = retime_trajectory_segment(
        encoder,
        positions,
        joint_map=mapping,
        original_duration=float(times[-1]),
        source_times=times,
        output_dt=args.output_dt,
        velocity_limits=args.velocity_limits,
        acceleration_limits=args.acceleration_limits,
        lead_speed=args.lead_speed,
        trail_speed=args.trail_speed,
        key=None if args.seed is None else jax.random.key(args.seed),
        config=RetimeConfig(starts=args.starts, iterations=args.iterations),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        positions=result.positions,
        velocities=result.velocities,
        accelerations=result.accelerations,
        times=result.times,
        dt=result.dt,
        robot_joint_names=np.asarray(names),
        model=np.asarray(encoder.spec.name),
        target_latent=result.target_latent,
    )
    print(
        f"Saved {args.output}; duration {times[-1]:.4g} -> {result.duration:.4g}s, "
        f"style cost {result.original_style_cost:.5g} -> {result.style_cost:.5g}"
    )


if __name__ == "__main__":
    main()
