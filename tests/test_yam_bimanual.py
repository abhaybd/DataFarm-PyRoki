"""Joint encoder objective, cross-arm geometry derivatives and common planning frame."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("pyroki")
pytest.importorskip("av")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def examples(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))


def test_independent_adapter_is_mean_cost_in_planning_and_retiming(encoder, motion):
    from _yam_bimanual import IndependentMotionEncoder

    from datafarm_pyroki import EncoderSpec

    names = encoder.spec.joint_names
    left = replace(encoder, spec=EncoderSpec("left", tuple("left_" + n for n in names), 15))
    right = replace(encoder, spec=EncoderSpec("right", tuple("right_" + n for n in names), 15))
    both = IndependentMotionEncoder(left, right)
    with pytest.raises(ValueError, match="rates"):
        IndependentMotionEncoder(
            left, replace(right, spec=EncoderSpec("right", right.spec.joint_names, 30))
        )
    q = jnp.asarray(np.concatenate([motion, motion * 0.8], axis=1))
    cost, grad = jax.jit(jax.value_and_grad(lambda q: both.cost(q, 1 / 15)))(q)
    expected = (left.cost(q[:, :7], 1 / 15) + right.cost(q[:, 7:], 1 / 15)) / 2
    np.testing.assert_allclose(cost, expected, rtol=1e-6)
    assert np.isfinite(grad).all() and np.linalg.norm(grad[:, :7]) > 0
    assert np.linalg.norm(grad[:, 7:]) > 0
    z, _ = both.encode(both.features_at_model_rate(q[None]))
    np.testing.assert_allclose(
        jnp.sum(((z[0] - both.latent_mean) @ both.precision_cholesky) ** 2), cost, rtol=1e-6
    )


@pytest.fixture
def scene(tmp_path):
    from _yam_handoff import scene_settings
    from _yam_scene import MENAGERIE_COMMIT, build_scene, initial_state

    assets = ROOT / "work/yam-menagerie" / MENAGERIE_COMMIT / "i2rt_yam"
    if not (assets / "yam_linear_4310.xml").exists():
        pytest.skip("Populate the optional YAM Menagerie cache by running the demo")
    model = build_scene(assets, tmp_path / "scene.xml", **scene_settings())
    return model, initial_state(model, settle=False)


def test_combined_native_fk_and_joint_order(scene, tmp_path):
    import pyroki as pk
    import yourdfpy
    from _yam_scene import ARMS, HOME, JOINT_NAMES, make_bimanual_urdf, make_urdf

    model, data = scene
    for arm in ARMS:
        make_urdf(model, data, arm, tmp_path / f"{arm}.urdf")
    path = tmp_path / "both.urdf"
    make_bimanual_urdf(tmp_path, path)
    robot = pk.Robot.from_urdf(yourdfpy.URDF.load(path))
    assert tuple(robot.joints.actuated_names) == JOINT_NAMES
    for q in np.tile(HOME, 2) + np.random.default_rng(14).normal(size=(3, 12)) * 0.06:
        data.qpos[[model.joint(n).qposadr[0] for n in JOINT_NAMES]] = q
        mujoco.mj_kinematics(model, data)
        poses = np.asarray(robot.forward_kinematics(jnp.asarray(q)))
        for arm in ARMS:
            np.testing.assert_allclose(
                poses[robot.links.names.index(f"{arm}_tcp"), 4:],
                data.site_xpos[model.site(f"{arm}_grasp_site").id],
                atol=3e-6,
            )


def test_coupled_payload_distance_differentiates_both_arms(scene):
    from _yam_collision import SceneDistances
    from _yam_scene import HOME

    model, data = scene
    geometry = SceneDistances(model, "left", coupled=True)
    q = np.tile(HOME, 2) + np.array([0.05, 0.03, -0.02, 0.01, -0.02, 0.01] * 2)
    data.qpos[geometry.qpos] = q
    mujoco.mj_kinematics(model, data)
    adr = model.joint("cube0_free").qposadr[0]
    # A deliberately extended payload transform makes its separation from the
    # left fingers depend on both chains, exercising both witness Jacobians.
    data.qpos[adr : adr + 3] = data.site_xpos[model.site("left_grasp_site").id] + [
        0.02,
        0.035,
        0.005,
    ]
    state = geometry.context(data, carried_cube=0, carrier_arm="right")
    _, analytic = geometry.evaluate(q, state)
    eps = 1e-5
    numeric = np.stack(
        [
            (
                geometry.evaluate(q + direction * eps, state)[0]
                - geometry.evaluate(q - direction * eps, state)[0]
            )
            / (2 * eps)
            for direction in np.eye(12)
        ],
        axis=1,
    )
    np.testing.assert_allclose(analytic[2], numeric[2], atol=2e-6)
    assert np.linalg.norm(analytic[2, :6]) > 1e-3
    assert np.linalg.norm(analytic[2, 6:]) > 1e-3


def test_three_panel_video_speedup_and_final_frame_hold(tmp_path):
    import imageio.v3 as iio
    from yam_bimanual import stack_videos

    paths = []
    for i, count in enumerate((2, 3, 5)):
        path = tmp_path / f"{i}.mp4"
        paths.append(path)
        with iio.imopen(path, "w", plugin="pyav") as video:
            video.init_video_stream("libx264", fps=30, pixel_format="yuv420p")
            for j in range(count):
                video.write_frame(np.full((128, 200, 3), 30 + i * 50 + j * 10, np.uint8))
    destination = tmp_path / "comparison.mp4"
    stack_videos(paths, destination, speedup=2)
    with iio.imopen(destination, "r", plugin="pyav") as video:
        frames = list(video.iter())
    assert len(frames) == 3 and frames[0].shape == (128, 600, 3)
    np.testing.assert_allclose(
        [frames[-1][80, i * 200 + 100, 0] for i in range(3)], [40, 100, 170], atol=5
    )
