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


@pytest.mark.parametrize("direct", [False, True])
def test_handoff_barriers_survive_unequal_arm_timing(direct):
    from _yam_handoff import STARTS, TARGETS, plan_handoffs
    from _yam_scene import ARMS

    assert np.all(STARTS[:, 1] > 0) and np.all(TARGETS[:, 1] < 0)

    def joint(name):
        if name.startswith("cube"):
            index = 40 + 7 * int(name[4])
        else:
            arm = 0 if name.startswith("left_") else 1
            tail = name.split("_", 1)[1]
            index = arm * 8 + (
                int(tail[5:]) - 1 if tail.startswith("joint") else 6 + (tail.startswith("right_"))
            )
        return SimpleNamespace(qposadr=[index])

    model = SimpleNamespace(joint=joint)
    labels = []

    def leg(a, b, label, **kwargs):
        labels.append(label)
        return np.stack([a, b])

    planners = {
        arm: SimpleNamespace(
            arm=arm,
            model=model,
            scratch=SimpleNamespace(qpos=np.zeros(80)),
            ik_records=[],
            leg_records=[],
            endpoint=lambda xyz, current, cube, orientation: np.r_[xyz, np.zeros(3)],
            leg=leg,
        )
        for arm in ARMS
    }
    if direct:
        for planner in planners.values():
            planner.collisions = SimpleNamespace(
                evaluate=lambda *args: None,
                validate_path=lambda *args: 0.0,
                data=planner.scratch,
            )
            planner.last_collision_state = None

    def timeline(planner, steps, args):
        # Receiver is deliberately slower; all gripper events must wait for it.
        duration = len(steps) * (3.0 if planner.arm == "right" else 2.0)
        return SimpleNamespace(
            positions=np.stack([steps[0].positions[0], steps[-1].positions[-1]]),
            times=np.array([0.0, duration]),
        ), []

    q, t, events, records = plan_handoffs(
        planners, SimpleNamespace(direct_transfers=direct), timeline
    )
    assert q.shape[1] == 12 and np.all(np.diff(t) > 0)
    assert len(events) > 0
    if direct:
        assert not any(
            suffix in label
            for label in labels
            for suffix in (
                "giver approach",
                "giver descend",
                "giver lift",
                "receiver transport",
                "receiver lower",
                "retreat",
            )
        )
        assert sum("giver grasp directly" in label for label in labels) == 3
        assert sum("receiver places directly" in label for label in labels) == 3
    for cube in range(3):
        commands = [e for e in events if e["command"].get("cube") == cube]
        assert [(e["command"]["arm"], e["command"]["action"]) for e in commands] == [
            ("left", "close"),
            ("right", "close"),
            ("left", "open"),
            ("right", "open"),
        ]
        assert commands[2]["time"] >= commands[1]["time"] + 0.65 - 1e-9
        receive = next(
            p
            for p in records["phases"]
            if p["cube"] == cube and p["label"] == "receiver approaches"
        )
        assert commands[1]["time"] >= receive["start_s"] + receive["duration_s"] - 1e-9
        # Giver remains stationary throughout receiver acquisition.
        rows = (t >= receive["start_s"] + 0.01) & (t <= receive["start_s"] + receive["duration_s"])
        np.testing.assert_allclose(q[rows, :6], np.broadcast_to(q[rows, :6][0], q[rows, :6].shape))


def test_shared_phase_dilation_preserves_geometry_and_meets_motion_caps():
    from _yam_handoff import enforce_phase_limits

    t = np.linspace(0, 0.1, 101)
    q = np.tile((0.5 * (1 - np.cos(np.pi * t / t[-1])))[:, None], (1, 12))
    stretched, scale = enforce_phase_limits(q, t)
    assert scale > 1 and stretched[-1] > t[-1]
    velocity = np.gradient(q, stretched, axis=0)
    acceleration = np.gradient(velocity, stretched, axis=0)
    assert np.abs(velocity).max() <= 2 and np.abs(acceleration).max() <= 20


def test_giver_release_is_blocked_without_receiver_contact(tmp_path):
    from _yam_handoff import TARGETS, scene_settings
    from _yam_scene import HOME, MENAGERIE_COMMIT, build_scene
    from _yam_simulation import simulate

    assets = ROOT / "work/yam-menagerie" / MENAGERIE_COMMIT / "i2rt_yam"
    if not (assets / "yam_linear_4310.xml").exists():
        pytest.skip("Run the demo to populate optional YAM assets")
    model = build_scene(assets, tmp_path / "scene.xml", **scene_settings())
    events = [dict(time=0.0, command=dict(arm="left", action="open", cube=0, handoff=True))]
    with pytest.raises(RuntimeError, match="giver release blocked"):
        simulate(
            model,
            np.tile(np.r_[HOME, HOME], (2, 1)),
            np.array([0.0, 0.02]),
            events,
            tmp_path,
            enabled=False,
            retimed=False,
            render=False,
            targets=TARGETS,
            handoff=True,
        )
    assert (tmp_path / "handoff_failure.npz").exists()
