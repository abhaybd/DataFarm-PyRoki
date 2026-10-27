"""Demo timing settings and diagnostics around the shared trajectory API."""

import time
from dataclasses import asdict

import numpy as np

from datafarm_pyroki import (
    RetimeConfig,
    RetimedTrajectorySegment,
    interpolate_plan,
    retime_plan,
    sample_plan,
)


def make_timeline(planner, steps, args):
    before = time.monotonic()
    if not planner.datafarm or not args.retime:
        plan = sample_plan(steps)
        events = [{"time": e.time, "command": e.event.command} for e in plan.events]
        return plan.positions, plan.times, events, [], 0.0
    print("Retiming gripper-delimited trajectory segments with the frozen encoder…", flush=True)
    config = RetimeConfig(
        knots=16,
        starts=args.retime_starts,
        iterations=args.retime_iterations,
        min_duration_multiplier=0.7,
        max_duration_multiplier=3.0,
    )
    plan = retime_plan(
        planner.encoder,
        interpolate_plan(steps),
        joint_map=planner.mapping,
        output_dt=0.02,
        velocity_limits=np.full(7, 2.0),
        acceleration_limits=np.full(7, 20.0),
        boundary_speed=0.0,
        config=config,
    )
    trajectory_segments = [
        step for step in plan.steps if isinstance(step, RetimedTrajectorySegment)
    ]
    records = [
        {
            "duration": s.duration,
            "style_before": s.original_style_cost,
            "style_after": s.style_cost,
            "regularization_cost": s.regularization_cost,
            "motion_cost": s.motion_cost,
            "objective": s.objective,
            "settings": asdict(config),
        }
        for s in trajectory_segments
    ]
    events = [{"time": event.time, "command": event.event.command} for event in plan.events]
    return plan.positions, plan.times, events, records, time.monotonic() - before
