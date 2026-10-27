"""Trajectory utilities use explicit coordinates and clocks, without a simulator."""

import numpy as np
import pytest

from datafarm_pyroki import (
    GripperEvent,
    MotionLeg,
    TrajectoryValidationError,
    interpolate_plan,
    interpolate_trajectory,
    sample_plan,
    validate_trajectory,
)


def test_fixed_clock_preserves_payloads_holds_and_three_joint_start():
    home = np.array([0.2, -0.3, 0.4])
    goal = home + [0.05, 0.0, 0.0]
    command = {"width": 0.0}
    before = GripperEvent({"width": 0.1}, dwell_seconds=0.04)
    close = GripperEvent(command, dwell_seconds=0.2)
    result = sample_plan(
        [
            before,
            MotionLeg(np.stack([home, goal]), 0.5),
            close,
            MotionLeg(np.stack([goal, home]), 0.25),
        ]
    )
    assert result.events[0].event is before
    assert result.events[1].event.command is command
    np.testing.assert_allclose(result.events[1].time, 0.54)
    np.testing.assert_allclose(result.times[-1], 0.99)
    np.testing.assert_array_equal(result.positions[[0, -1]], np.stack([home, home]))
    hold = (result.times >= 0.54) & (result.times <= 0.74)
    assert hold.sum() > 2
    np.testing.assert_allclose(
        result.positions[hold], np.broadcast_to(goal, result.positions[hold].shape)
    )
    assert np.all(np.diff(result.times) > 0)


def test_fixed_clock_flows_through_joins_and_eases_at_gripper_boundaries():
    home = np.array([0.3, -0.5])
    delta = np.array([0.1, 0.0])
    result = sample_plan(
        [
            MotionLeg(np.linspace(home, home + delta, 6), 0.1),
            MotionLeg(np.linspace(home + delta, home + 2 * delta, 6), 0.1),
        ]
    )
    velocity = np.diff(result.positions[:, 0]) / np.diff(result.times)
    middle = np.searchsorted(result.times, 0.5)
    assert np.all(velocity > 0)
    assert min(velocity[middle - 1 : middle + 1]) > 0.15
    assert velocity[0] < 0.1 and velocity[-1] < 0.1
    np.testing.assert_allclose(result.positions[middle], home + delta)


def test_event_only_clock_requires_explicit_start_and_retains_zero_duration_events():
    event = GripperEvent("close", dwell_seconds=0.04)
    zero = GripperEvent("open")
    with pytest.raises(ValueError, match="initial_position"):
        sample_plan([event])
    result = sample_plan([event, zero], initial_position=[1.0, 2.0])
    np.testing.assert_array_equal(result.positions, [[1, 2], [1, 2], [1, 2]])
    assert result.events[0].event is event and result.events[1].event is zero
    assert result.times[-1] == 0.04 and np.all(np.diff(result.times) > 0)


def test_interpolation_preserves_nonuniform_source_clock_and_events():
    q = np.array([[0.0, 0.2], [0.1, 0.1], [0.2, 0.0]])
    result = interpolate_trajectory(q, [2.0, 2.1, 2.4], output_dt=0.05)
    np.testing.assert_allclose(result.times[-1], 0.4)
    np.testing.assert_array_equal(result.positions[[0, -1]], q[[0, -1]])
    event = GripperEvent("close")
    steps = interpolate_plan([MotionLeg(q[:2], 0.1, "a"), MotionLeg(q[1:], 0.3, "b"), event])
    assert steps[-1] is event
    assert steps[0].label == "a; b"
    np.testing.assert_allclose((len(steps[0].positions) - 1) * steps[0].dt, 0.4)


def test_validation_catches_interval_speed_even_with_rest_endpoints():
    with pytest.raises(TrajectoryValidationError, match="velocity"):
        validate_trajectory(
            [[0.0], [1.0]], [0.0, 0.1], velocity_limits=[2.0], acceleration_limits=[100.0]
        )
    with pytest.raises(TrajectoryValidationError, match="increasing"):
        validate_trajectory(
            [[0.0], [0.0]], [0.0, 0.0], velocity_limits=[2.0], acceleration_limits=[100.0]
        )
    with pytest.raises(TrajectoryValidationError, match="position"):
        validate_trajectory(
            [[0.0], [0.2]],
            [0.0, 1.0],
            velocity_limits=[2.0],
            acceleration_limits=[100.0],
            lower_limits=[-0.1],
            upper_limits=[0.1],
        )
    v, a = validate_trajectory(
        [[0.0, 0.0], [0.02, 0.01], [0.03, 0.015]],
        [0.0, 0.1, 0.3],
        velocity_limits=[2.0, 2.0],
        acceleration_limits=[10.0, 10.0],
    )
    assert v.shape == a.shape == (3, 2)
    np.testing.assert_array_equal(v[[0, -1]], np.zeros((2, 2)))


@pytest.mark.parametrize("dt", [0, -0.1, np.nan])
def test_interpolation_rejects_invalid_clocks(dt):
    with pytest.raises(ValueError, match="output_dt"):
        interpolate_trajectory([[0.0], [1.0]], [0.0, 1.0], output_dt=dt)
