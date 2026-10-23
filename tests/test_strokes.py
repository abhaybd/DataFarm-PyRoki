from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from datafarm_pyroki import (
    GripperEvent,
    JointMap,
    MotionLeg,
    RetimeConfig,
    RetimedStroke,
    RetimingError,
    Stroke,
    group_strokes,
    retime_plan,
)
from datafarm_pyroki.strokes import _stitch_plan


def plan_input():
    t = np.linspace(0.0, 1.0, 16)[:, None]
    q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]) + 0.03 * np.sin(t * np.arange(1, 8)[None])
    before = GripperEvent({"width": 0.08}, gripper="first", dwell_seconds=0.04)
    middle = GripperEvent({"width": 0.0}, gripper="second", dwell_seconds=0.08)
    after = GripperEvent({"width": 0.08}, gripper="first", dwell_seconds=0.06)
    return q, [
        before,
        MotionLeg(q[:9], 0.1, "retract"),
        MotionLeg(q[8:], 0.15, "transit"),
        middle,
        MotionLeg(q[::-1], 0.1, "return"),
        after,
    ]


def test_grouping_preserves_events_and_nonuniform_source_clock():
    q, steps = plan_input()
    grouped = group_strokes(steps)
    assert len(grouped) == 5
    assert grouped[0] is steps[0] and grouped[2] is steps[3] and grouped[4] is steps[5]
    assert isinstance(grouped[1], Stroke)
    np.testing.assert_array_equal(grouped[1].positions, q)
    np.testing.assert_allclose(np.diff(grouped[1].times), [0.1] * 8 + [0.15] * 7)
    assert grouped[1].labels == ("retract", "transit")
    assert group_strokes([steps[0]]) == (steps[0],)
    with pytest.raises(ValueError, match="Discontinuous"):
        group_strokes([MotionLeg(q, 0.1), steps[3], MotionLeg(q + 1.0, 0.1)])


def test_actual_plan_retiming_preserves_commands_holds_and_limits(encoder):
    q, steps = plan_input()
    mapping = JointMap.from_names(encoder.spec.joint_names, encoder.spec.joint_names)
    plan = retime_plan(
        encoder,
        steps,
        joint_map=mapping,
        output_dt=0.04,
        velocity_limits=np.full(7, 10.0),
        acceleration_limits=np.full(7, 1000.0),
        config=RetimeConfig(starts=1, iterations=2, knots=8),
    )
    assert len(plan.steps) == 5 and len(plan.events) == 3
    assert [e.event for e in plan.events] == [steps[0], steps[3], steps[5]]
    assert all(
        e.event.command is original.command
        for e, original in zip(plan.events, [steps[0], steps[3], steps[5]], strict=True)
    )
    assert plan.events[0].time == 0.0
    np.testing.assert_allclose(plan.events[1].time, 0.04 + plan.steps[1].duration)
    assert np.all(np.diff(plan.times) > 0)
    np.testing.assert_array_equal(plan.positions[[0, -1]], q[[0, 0]])
    assert np.max(abs(plan.velocities)) <= 10 and np.max(abs(plan.accelerations)) <= 1000
    for event in plan.events:
        hold = (plan.times >= event.time - 1e-10) & (
            plan.times <= event.time + event.event.dwell_seconds + 1e-10
        )
        assert hold.sum() >= 2
        assert np.max(np.ptp(plan.positions[hold], axis=0)) < 1e-12


def test_stitched_reversal_is_checked_even_if_individual_strokes_pass():
    # Each standalone stroke has constant velocity and zero acceleration. Their
    # reversal must still be rejected by the full-plan check.
    times = np.array([0.0, 0.05, 0.1])
    q = np.array([[0.0], [0.05], [0.1]])
    forward = RetimedStroke(
        q,
        np.ones_like(q),
        np.zeros_like(q),
        times,
        0.1,
        0.05,
        0.0,
        0.0,
        0.0,
        0.0,
        np.zeros(1),
        q,
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 0.5, 1.0]),
        1.0,
        1.0,
    )
    backward = replace(forward, positions=q[::-1], velocities=-forward.velocities)
    with pytest.raises(RetimingError, match="junction"):
        _stitch_plan(
            [forward, GripperEvent("close"), backward],
            np.array([2.0]),
            np.array([1.0]),
            output_dt=0.05,
        )


def test_no_motion_and_discontinuous_plan_errors(encoder):
    mapping = JointMap.from_names(encoder.spec.joint_names, encoder.spec.joint_names)
    with pytest.raises(ValueError, match="at least one motion"):
        retime_plan(
            encoder,
            [GripperEvent("open")],
            joint_map=mapping,
            output_dt=0.1,
            velocity_limits=jnp.ones(7),
            acceleration_limits=jnp.ones(7),
        )
