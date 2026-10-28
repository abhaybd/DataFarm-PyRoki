"""Two-arm example timing, task sequence, and simulator/planner coordinate contract."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("pyroki")
pytest.importorskip("av")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def examples(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))


def test_independent_arm_clocks_preserve_events_and_hold_finished_arm():
    from yam_pick_place import combine

    from datafarm_pyroki import GripperEvent
    from datafarm_pyroki.trajectory_segments import TimedGripperEvent

    left = SimpleNamespace(
        positions=np.array([np.zeros(6), np.ones(6)]),
        times=np.array([0.0, 1.0]),
        events=[
            TimedGripperEvent(0.5, GripperEvent({"arm": "left", "action": "close"}, gripper="left"))
        ],
    )
    right = SimpleNamespace(
        positions=np.array([np.zeros(6), np.full(6, 2.0)]),
        times=np.array([0.0, 2.0]),
        events=[
            TimedGripperEvent(
                1.5, GripperEvent({"arm": "right", "action": "close"}, gripper="right")
            )
        ],
    )
    q, times, events = combine([left, right])
    assert q.shape == (101, 12)
    np.testing.assert_allclose(q[:, :6], np.minimum(times, 1.0)[:, None] * np.ones((1, 6)))
    np.testing.assert_allclose(q[:, 6:], times[:, None] * np.ones((1, 6)))
    assert [(e["time"], e["command"]["arm"]) for e in events] == [(0.5, "left"), (1.5, "right")]


@pytest.mark.parametrize("arm,cubes", [("left", [0, 1]), ("right", [2, 3])])
def test_each_arm_uses_own_cubes_and_gripper(arm, cubes):
    from _yam_planning import plan_arm

    from datafarm_pyroki import GripperEvent, MotionLeg

    contacts = []

    def leg(start, goal, label, *, cube=None, contact=False):
        contacts.append(contact)
        return np.stack([start, goal])

    planner = SimpleNamespace(
        arm=arm,
        endpoint=lambda xyz, current, cube: np.r_[xyz, np.zeros(3)],
        leg=leg,
        model=SimpleNamespace(joint=lambda n: SimpleNamespace(qposadr=[int(n[4]) * 7])),
        scratch=SimpleNamespace(qpos=np.zeros(28)),
    )
    steps = plan_arm(planner)
    assert sum(isinstance(s, MotionLeg) for s in steps) == 13
    assert contacts == [False, True, True, False, True, True] * 2 + [False]
    events = [s for s in steps if isinstance(s, GripperEvent)]
    assert all(e.gripper == arm and e.command["arm"] == arm for e in events)
    assert [e.command["cube"] for e in events] == [c for c in cubes for _ in range(2)]


def test_native_both_arm_fk_and_unattached_cubes(tmp_path):
    import jax.numpy as jnp
    import pyroki as pk
    import yourdfpy
    from _yam_scene import (
        ARM_NAMES,
        ARMS,
        HOME,
        MENAGERIE_COMMIT,
        build_scene,
        initial_state,
        make_urdf,
    )

    assets = ROOT / "work/yam-menagerie" / MENAGERIE_COMMIT / "i2rt_yam"
    if not (assets / "yam_linear_4310.xml").exists():
        pytest.skip("Run yam_pick_place once to populate its optional Menagerie asset cache")
    model = build_scene(assets, tmp_path / "scene.xml")
    data = initial_state(model, settle=False)
    assert not np.any(model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    for i in range(4):
        assert model.joint(f"cube{i}_free").type == mujoco.mjtJoint.mjJNT_FREE
    for arm in ARMS:
        path = tmp_path / f"{arm}.urdf"
        make_urdf(model, data, arm, path)
        robot = pk.Robot.from_urdf(yourdfpy.URDF.load(path))
        assert tuple(robot.joints.actuated_names) == ARM_NAMES[arm]
        tcp = robot.links.names.index("tcp")
        for q in HOME + np.random.default_rng(4).normal(size=(3, 6)) * 0.08:
            for name, value in zip(ARM_NAMES[arm], q, strict=True):
                data.qpos[model.joint(name).qposadr] = value
            mujoco.mj_forward(model, data)
            actual = np.asarray(robot.forward_kinematics(jnp.asarray(q)))[tcp, 4:]
            np.testing.assert_allclose(
                actual, data.site_xpos[model.site(arm + "_grasp_site").id], atol=3e-6
            )
