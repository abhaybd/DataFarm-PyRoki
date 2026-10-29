"""Contact-driven left-to-right transfers with shared barriers between marginal planners."""

from __future__ import annotations

import numpy as np
from _yam_scene import ARM_NAMES, ARMS, HOME

from datafarm_pyroki import MotionLeg

STARTS = np.array([[0.27, 0.22, 0.021], [0.31, 0.34, 0.021], [0.34, 0.44, 0.021]])
TARGETS = np.array([[0.39, -0.23, 0.037], [0.40, -0.34, 0.037], [0.33, -0.44, 0.037]])
CENTER = np.array([0.30, 0, 0.25])
DOWN = np.array([0.0, 0.0, 1.0, 0.0])
# Opposing tool axes; giver pinches vertically and receiver horizontally.
PRESENT = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
RECEIVE = np.array([0.5, -0.5, 0.5, 0.5])


def scene_settings():
    return dict(
        starts=STARTS, targets=TARGETS, tray_center=(0.37, -0.34), tray_halfsize=(0.10, 0.18)
    )


def enforce_phase_limits(q, times):
    """Uniformly dilate a shared phase if interpolation exceeds the common motion caps."""
    velocity = np.gradient(q, times, axis=0)
    acceleration = np.gradient(velocity, times, axis=0)
    scale = max(
        1.0, float(np.abs(velocity).max() / 2), float(np.sqrt(np.abs(acceleration).max() / 20))
    )
    if scale > 1:
        scale *= 1.02
    return times * scale, scale


def plan_handoffs(planners, args, timeline_fn, *, joint_planner=None):
    """Optimize each arm with its own model; synchronize before every ownership change.

    The receiver must reach its grasp, close and dwell before the giver opens.
    Retiming changes phase durations, never the event/barrier sequence.
    """
    current = {arm: HOME.copy() for arm in ARMS}
    closed = {arm: False for arm in ARMS}
    held = {arm: None for arm in ARMS}
    direct = bool(getattr(args, "direct_transfers", False))
    chunks, timestamps, events, phases = [], [], [], []
    cursor = 0.0
    arm_retiming = {arm: [] for arm in ARMS}

    def scratch():
        for planner in planners.values():
            for arm in ARMS:
                for name, value in zip(ARM_NAMES[arm], current[arm], strict=True):
                    planner.scratch.qpos[planner.model.joint(name).qposadr] = value
                for side in ("left", "right"):
                    planner.scratch.qpos[planner.model.joint(f"{arm}_{side}_finger").qposadr] = (
                        0.02 if closed[arm] else 0.0475
                    )

    def object_at(cube, xyz, quaternion=None):
        for planner in planners.values():
            adr = planner.model.joint(f"cube{cube}_free").qposadr[0]
            planner.scratch.qpos[adr : adr + 3] = xyz
            if quaternion is not None:
                planner.scratch.qpos[adr + 3 : adr + 7] = quaternion

    def append(q, t):
        nonlocal cursor
        if chunks:
            q, t = q[1:], t[1:]
        chunks.append(q)
        timestamps.append(t + cursor)
        cursor = float(timestamps[-1][-1])

    def phase(label, motions, cube):
        if joint_planner is not None:
            joint_phase(label, motions, cube)
            return
        before = {a: current[a].copy() for a in ARMS}
        timelines = {}
        phase_records = {}
        for arm, waypoints in motions.items():
            planner = planners[arm]
            steps = []
            for xyz, orientation, track, description in waypoints:
                scratch()
                if xyz is None:
                    goal = HOME.copy()
                else:
                    goal = planner.endpoint(xyz, current[arm], cube, orientation=orientation)
                name = f"cube {cube + 1}: {description}"
                if direct:
                    planner.carried_cube = held[arm]
                path = planner.leg(current[arm], goal, name, cube=cube, contact=track)
                if direct and held[arm] is not None:
                    # Carry ideal geometry between legs. Replay uses free cubes.
                    planner.collisions.evaluate(path[-1], planner.last_collision_state)
                    adr = planner.model.joint(f"cube{cube}_free").qposadr[0]
                    pose = planner.collisions.data.qpos[adr : adr + 7].copy()
                    object_at(cube, pose[:3], pose[3:])
                steps.append(MotionLeg(path, 0.1, name))
                current[arm] = goal
            timeline, records = timeline_fn(planner, steps, args)
            timelines[arm] = timeline
            arm_retiming[arm].extend(records)
            phase_records[arm] = dict(duration_s=float(timeline.times[-1]), retiming=records)
            if direct:
                phase_records[arm]["interpolated_minimum_geometry_distance_m"] = (
                    planner.collisions.validate_path(
                        timeline.positions, planner.last_collision_state
                    )
                )
        duration = max(float(t.times[-1]) for t in timelines.values())
        local = np.linspace(0, duration, int(np.ceil(duration / 0.02)) + 1)
        positions = []
        for arm in ARMS:
            if arm in timelines:
                t = timelines[arm]
                positions.append(
                    np.stack([np.interp(local, t.times, c) for c in t.positions.T], axis=1)
                )
            else:
                positions.append(np.tile(before[arm], (len(local), 1)))
        q = np.concatenate(positions, axis=1)
        local, time_scale = enforce_phase_limits(q, local)
        duration = float(local[-1])
        events.append(
            dict(time=cursor, command=dict(action="phase", label=f"cube {cube + 1}: {label}"))
        )
        phases.append(
            dict(
                label=label,
                cube=cube,
                start_s=cursor,
                duration_s=duration,
                arms=phase_records,
                motion_limit_time_scale=time_scale,
            )
        )
        append(q, local)

    def joint_phase(label, motions, cube):
        """Use the same task barriers with a single 12-joint path and clock."""
        if not direct or any(len(steps) != 1 for steps in motions.values()):
            raise ValueError("Joint phases require direct transfers with one goal per moving arm")
        scratch()
        scene = planners["left"].scratch.qpos.copy()
        start = np.r_[current["left"], current["right"]]
        goals = {a: current[a].copy() for a in ARMS}
        tracking = []
        for arm in ARMS:
            if arm not in motions:
                tracking.append(False)
                continue
            xyz, orientation, track, _ = motions[arm][0]
            tracking.append(track)
            goals[arm] = (
                HOME.copy()
                if xyz is None
                else planners[arm].endpoint(xyz, current[arm], cube, orientation=orientation)
            )
        goal = np.r_[goals["left"], goals["right"]]
        path, record = joint_planner.solve(
            start,
            goal,
            scene,
            [a in motions for a in ARMS],
            tracking,
            held,
            cube,
            f"cube {cube + 1}: {label}",
        )
        timeline, retiming = timeline_fn(joint_planner, [MotionLeg(path, 0.1, label)], args)
        minimum = joint_planner.validate(timeline.positions)
        local, scale = enforce_phase_limits(timeline.positions, timeline.times)
        duration = float(local[-1])
        events.append(
            dict(time=cursor, command=dict(action="phase", label=f"cube {cube + 1}: {label}"))
        )
        phases.append(
            dict(
                label=label,
                cube=cube,
                start_s=cursor,
                duration_s=duration,
                arms={a: dict(duration_s=duration, retiming=retiming) for a in motions},
                joint=record,
                shared_retiming=retiming,
                minimum_geometry_distance_m=minimum,
                motion_limit_time_scale=scale,
            )
        )
        append(timeline.positions, local)
        for arm in ARMS:
            current[arm] = goals[arm]
            if arm in motions:
                arm_retiming[arm].extend(retiming)
        if any(c is not None for c in held.values()):
            pose = joint_planner.payload_pose(path[-1], cube)
            object_at(cube, pose[:3], pose[3:])

    def grip(arm, action, cube, dwell, *, handoff=False):
        closed[arm] = action == "close"
        held[arm] = cube if closed[arm] else None
        events.append(
            dict(time=cursor, command=dict(arm=arm, action=action, cube=cube, handoff=handoff))
        )
        q = np.r_[current["left"], current["right"]]
        append(
            np.tile(q, (int(np.ceil(dwell / 0.02)) + 1, 1)),
            np.linspace(0, dwell, int(np.ceil(dwell / 0.02)) + 1),
        )

    ready = CENTER + np.array([0.0, -0.10, 0.0])
    withdrawn = CENTER + np.array([0.0, 0.10, 0.0])
    for cube, (start, target) in enumerate(zip(STARTS, TARGETS, strict=True)):
        print(f"Planning synchronized handoff {cube + 1}/{len(STARTS)}", flush=True)
        right = []
        if cube and not direct:
            previous = TARGETS[cube - 1]
            right.append((np.r_[previous[:2], 0.14], DOWN, True, "receiver retreat"))
        right.append((ready, RECEIVE, False, "receiver ready"))
        phase(
            "pick on source side",
            {
                "left": [
                    (np.r_[start[:2], 0.14], DOWN, False, "giver approach"),
                    (start, DOWN, True, "giver descend"),
                ]
                if not direct
                else [(start, DOWN, False, "giver grasp directly")],
                "right": right,
            },
            cube,
        )
        grip("left", "close", cube, 0.65)
        phase(
            "present at center",
            {
                "left": [
                    (np.r_[start[:2], 0.14], DOWN, True, "giver lift"),
                    (CENTER, PRESENT, False, "present for handoff"),
                ]
                if not direct
                else [(CENTER, PRESENT, False, "present directly")]
            },
            cube,
        )
        object_at(cube, CENTER)
        phase("receiver approaches", {"right": [(CENTER, RECEIVE, True, "receiver acquire")]}, cube)
        grip("right", "close", cube, 0.65, handoff=True)
        grip("left", "open", cube, 0.45, handoff=True)
        phase("giver withdraws", {"left": [(withdrawn, PRESENT, True, "giver withdraw")]}, cube)
        phase(
            "receiver places on target side",
            {
                "left": [(None, None, False, "giver return")],
                "right": [
                    (ready, RECEIVE, True, "receiver clear center"),
                    (np.r_[target[:2], 0.14], DOWN, False, "receiver transport"),
                    (target, DOWN, True, "receiver lower"),
                ]
                if not direct
                else [(target, DOWN, False, "receiver places directly")],
            },
            cube,
        )
        grip("right", "open", cube, 0.55)
        object_at(cube, target)
    last = len(STARTS) - 1
    phase(
        "complete",
        {
            "right": [
                (np.r_[TARGETS[-1, :2], 0.14], DOWN, True, "final retreat"),
                (None, None, False, "receiver return"),
            ]
            if not direct
            else [(None, None, False, "receiver return directly")]
        },
        last,
    )
    arms = {
        arm: dict(
            ik=p.ik_records, legs=p.leg_records, retiming=arm_retiming[arm], duration_s=cursor
        )
        for arm, p in planners.items()
    }
    return (
        np.concatenate(chunks),
        np.concatenate(timestamps),
        events,
        dict(arms=arms, phases=phases),
    )
