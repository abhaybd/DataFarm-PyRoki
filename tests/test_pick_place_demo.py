"""Optional simulation checks; install both planning and simulation extras."""

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("av")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def example_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))


@pytest.fixture
def scene():
    import _pick_place_scene

    return _pick_place_scene


@pytest.fixture
def runner():
    import _pick_place_demo

    return _pick_place_demo


def test_reuse_rejects_outdated_metadata_with_regeneration_hint(runner, scene, tmp_path):
    import json
    from types import SimpleNamespace

    output = tmp_path / "baseline"
    output.mkdir()
    np.savez_compressed(
        output / "plan.npz", positions=np.stack([scene.HOME, scene.HOME]), times=[0.0, 1.0]
    )
    (output / "planning.json").write_text(json.dumps({"free_space_constraints": "table clearance"}))
    args = SimpleNamespace(output=tmp_path, retime=False, reuse_plan=True)
    with pytest.raises(ValueError, match="rerun without --reuse-plan"):
        runner.run(args, False, None, tmp_path / "robot.urdf", plan_task=None)


def test_mujoco_pyroki_kinematics_and_free_cubes(scene, tmp_path):
    import jax.numpy as jnp
    import pyroki as pk
    import yourdfpy
    from _pick_place_scene import MENAGERIE_COMMIT

    assets = ROOT / "work/menagerie-demo" / MENAGERIE_COMMIT
    if not (assets / "download_complete.json").is_file():
        pytest.skip("Run the demo once to populate the optional Menagerie asset cache")
    model = scene.build_scene(assets, tmp_path / "scene.xml")
    data = scene.initial_state(model)
    scene.make_urdf(model, data, tmp_path / "robot.urdf")
    robot = pk.Robot.from_urdf(yourdfpy.URDF.load(tmp_path / "robot.urdf"))
    tcp = robot.links.names.index("tcp")
    for q in scene.HOME + np.random.default_rng(4).normal(size=(4, 7)) * 0.12:
        for i, value in enumerate(q, 1):
            data.qpos[model.joint(f"joint{i}").qposadr] = value
        mujoco.mj_forward(model, data)
        actual = np.asarray(robot.forward_kinematics(jnp.asarray(q)))[tcp, 4:]
        np.testing.assert_allclose(actual, data.site_xpos[model.site("rq_pinch").id], atol=3e-6)
    assert not np.any(model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    for i in range(3):
        assert model.joint(f"cube{i}_free").type == mujoco.mjtJoint.mjJNT_FREE
        assert data.xpos[model.body(f"cube{i}").id, 2] < scene.STARTS[i, 2]


def test_pick_place_sequence_preserves_contacts_and_gripper_events(scene):
    from types import SimpleNamespace

    import pick_place

    from datafarm_pyroki import GripperEvent, MotionLeg

    contacts = []

    def leg(start, goal, label, *, allowed_cube=None, track_pose=False):
        contacts.append(track_pose)
        return np.stack([start, goal])

    planner = SimpleNamespace(
        endpoint=lambda xyz, current, **kwargs: np.r_[xyz, np.zeros(4)],
        leg=leg,
        model=SimpleNamespace(joint=lambda name: SimpleNamespace(qposadr=[int(name[4]) * 7])),
        scratch=SimpleNamespace(qpos=np.zeros(21)),
    )
    steps = pick_place.plan_task(planner)
    assert sum(isinstance(step, MotionLeg) for step in steps) == 19
    assert contacts == [False, True, True, False, True, True] * 3 + [False]
    events = [step for step in steps if isinstance(step, GripperEvent)]
    assert [event.command for event in events] == [
        {"action": action, "cube": cube} for cube in range(3) for action in ("close", "open")
    ]
    np.testing.assert_array_equal(steps[0].positions[0], scene.HOME)
    np.testing.assert_array_equal(steps[-1].positions[-1], scene.HOME)
    for cube in range(3):
        np.testing.assert_array_equal(
            planner.scratch.qpos[cube * 7 : cube * 7 + 3], scene.TARGETS[cube]
        )


def test_video_encoding_after_jax_and_comparison_holds_final_frame(tmp_path):
    import imageio.v3 as iio
    import jax.numpy as jnp
    from _simulation import compare_videos

    # Rendering happens after planning has initialized JAX's worker threads.
    jnp.ones(2).block_until_ready()
    for mode, values in (("baseline", [30, 60]), ("datafarm", [100, 130, 160, 190])):
        directory = tmp_path / mode
        directory.mkdir()
        with iio.imopen(directory / "video.mp4", "w", plugin="pyav") as writer:
            writer.init_video_stream("libx264", fps=10, pixel_format="yuv420p")
            for value in values:
                writer.write_frame(np.full((32, 32, 3), value, dtype=np.uint8))

    compare_videos(
        tmp_path / "baseline/video.mp4",
        tmp_path / "datafarm/video.mp4",
        tmp_path / "comparison.mp4",
        fps=10,
    )
    frames = iio.imread(tmp_path / "comparison.mp4", plugin="pyav")
    assert frames.shape == (4, 32, 64, 3)
    assert iio.immeta(tmp_path / "comparison.mp4", plugin="pyav")["fps"] == 10
    np.testing.assert_allclose(frames[:, :, :32].mean(axis=(1, 2, 3)), [30, 60, 60, 60], atol=4)
    np.testing.assert_allclose(frames[:, :, 32:].mean(axis=(1, 2, 3)), [100, 130, 160, 190], atol=4)
