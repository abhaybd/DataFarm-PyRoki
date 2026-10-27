"""Shared playback uses explicit scene mappings and callbacks."""

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("av")


def test_playback_accepts_joint_mapping_events_and_local_time(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from _simulation import replay_trajectory

    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep="0.002" gravity="0 0 0"/>
    <worldbody>
      <body name="arm"><joint name="axis" type="slide"/><geom size="0.01"/></body>
      <body name="tool"><joint name="finger" type="slide"/><geom size="0.01"/></body>
    </worldbody>
    <actuator><position name="finger_motor" joint="finger" kp="10"/>
              <position name="arm_motor" joint="axis" kp="10"/></actuator>
    </mujoco>""")
    data = mujoco.MjData(model)
    data.time = 3.0
    source = np.array([[0.0], [0.01], [0.02]])
    command = {"position": 0.2}
    calls, observations = [], []

    def on_event(payload, state):
        calls.append(payload)
        state.ctrl[0] = payload["position"]

    result = replay_trajectory(
        model,
        data,
        source,
        [0.0, 0.02, 0.04],
        joint_names=["axis"],
        actuator_ids=[1],
        body_names=["arm"],
        events=[(0.01, command)],
        on_event=on_event,
        on_step=lambda t, state: observations.append(t),
        settle_seconds=0.02,
        log_dt=0.01,
        output=tmp_path,
        render=False,
    )
    assert len(calls) == 1 and calls[0] is command
    assert len(observations) >= 30
    assert result.duration == 0.06
    assert result.body_positions.shape == (len(result.times), 1, 3)
    assert (
        result.actual_positions.shape == result.commanded_positions.shape == (len(result.times), 1)
    )
    np.testing.assert_allclose(data.ctrl, [0.2, 0.02])
    np.testing.assert_allclose(source[:, 0], [0.0, 0.01, 0.02])
    assert np.all(np.diff(result.times) > 0) and result.times[-1] < 0.07
    with pytest.raises(ValueError, match="on_event"):
        replay_trajectory(
            model,
            data,
            source,
            [0, 0.02, 0.04],
            joint_names=["axis"],
            actuator_ids=[1],
            events=[(0.01, command)],
            output=tmp_path,
            render=False,
        )

    final_events = []
    replay_trajectory(
        model,
        data,
        source,
        [0.0, 0.02, 0.04],
        joint_names=["axis"],
        actuator_ids=[1],
        events=[(0.04, command)],
        on_event=lambda payload, state: final_events.append(payload),
        settle_seconds=0.0,
        output=tmp_path,
        render=False,
    )
    assert len(final_events) == 1 and final_events[0] is command
