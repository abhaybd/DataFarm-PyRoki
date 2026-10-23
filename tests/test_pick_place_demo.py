"""Optional simulation checks; install both planning and simulation extras."""

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("imageio_ffmpeg")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    import pick_place

    return pick_place


def test_baseline_clock_preserves_gripper_events_and_holds(demo):
    from datafarm_pyroki import GripperEvent, MotionLeg

    home = demo.HOME.copy()
    goal = home + np.array([0.05, 0, 0, 0, 0, 0, 0])
    close = {"action": "close", "cube": 0}
    steps = [
        MotionLeg(np.stack([home, goal]), 0.5),
        GripperEvent(close, dwell_seconds=0.2),
        MotionLeg(np.stack([goal, home]), 0.25),
    ]
    q, times, events = demo.baseline_timeline(steps)
    assert events == [{"time": 0.5, "command": close}]
    assert events[0]["command"] is close
    np.testing.assert_allclose(times[-1], 0.95)
    np.testing.assert_array_equal(q[[0, -1]], np.stack([home, home]))
    hold = (times >= 0.5) & (times <= 0.7)
    assert hold.sum() > 2
    np.testing.assert_allclose(q[hold], np.broadcast_to(goal, q[hold].shape))


def test_baseline_flows_through_leg_joins_and_eases_only_at_stroke_ends(demo):
    from datafarm_pyroki import MotionLeg

    home = demo.HOME.copy()
    delta = np.array([0.1, 0, 0, 0, 0, 0, 0])
    steps = [
        MotionLeg(np.linspace(home, home + delta, 6), 0.1),
        MotionLeg(np.linspace(home + delta, home + 2 * delta, 6), 0.1),
    ]
    q, times, events = demo.baseline_timeline(steps)
    assert events == []
    velocity = np.diff(q[:, 0]) / np.diff(times)
    assert np.all(velocity > 0)
    middle = np.searchsorted(times, 0.5)
    assert min(velocity[middle - 1 : middle + 1]) > 0.15
    assert velocity[0] < 0.1 and velocity[-1] < 0.1
    np.testing.assert_allclose(q[middle], home + delta)


def test_mujoco_pyroki_kinematics_and_free_cubes(demo, tmp_path):
    import jax.numpy as jnp
    import pyroki as pk
    import yourdfpy
    from _pick_place_scene import MENAGERIE_COMMIT

    assets = ROOT / "work/menagerie-demo" / MENAGERIE_COMMIT
    if not (assets / "download_complete.json").is_file():
        pytest.skip("Run the demo once to populate the optional Menagerie asset cache")
    model = demo.build_scene(assets, tmp_path / "scene.xml")
    data = demo.initial_state(model)
    demo.make_urdf(model, data, tmp_path / "robot.urdf")
    robot = pk.Robot.from_urdf(yourdfpy.URDF.load(tmp_path / "robot.urdf"))
    tcp = robot.links.names.index("tcp")
    for q in demo.HOME + np.random.default_rng(4).normal(size=(4, 7)) * 0.12:
        for i, value in enumerate(q, 1):
            data.qpos[model.joint(f"joint{i}").qposadr] = value
        mujoco.mj_forward(model, data)
        actual = np.asarray(robot.forward_kinematics(jnp.asarray(q)))[tcp, 4:]
        np.testing.assert_allclose(actual, data.site_xpos[model.site("rq_pinch").id], atol=3e-6)
    assert not np.any(model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    for i in range(3):
        assert model.joint(f"cube{i}_free").type == mujoco.mjtJoint.mjJNT_FREE
        assert data.xpos[model.body(f"cube{i}").id, 2] < demo.STARTS[i, 2]
