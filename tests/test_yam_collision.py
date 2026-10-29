"""Actual collision geometry derivatives, attachments, and contact exceptions."""

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("pyroki")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    from _yam_collision import SceneDistances
    from _yam_handoff import scene_settings
    from _yam_scene import HOME, MENAGERIE_COMMIT, build_scene, initial_state

    assets = ROOT / "work/yam-menagerie" / MENAGERIE_COMMIT / "i2rt_yam"
    if not (assets / "yam_linear_4310.xml").exists():
        pytest.skip("Populate the optional YAM asset cache by running the demo")
    model = build_scene(assets, tmp_path / "scene.xml", **scene_settings())
    data = initial_state(model, settle=False)
    collision = SceneDistances(model, "left")
    q = HOME + np.array([0.05, 0.03, -0.02, 0.01, -0.02, 0.01])
    data.qpos[collision.qpos] = q
    mujoco.mj_forward(model, data)
    return model, data, collision, q


@pytest.mark.parametrize("offset", [[0.04, 0.01, 0.005], [0.02, 0.035, 0.005]])
def test_distance_gradient_matches_finite_difference_outside_and_inside(scene, offset):
    import jax
    import jax.numpy as jnp

    model, data, collision, q = scene
    adr = model.joint("cube1_free").qposadr[0]
    data.qpos[adr : adr + 3] = data.site_xpos[collision.site] + offset
    state = collision.context(data, allowed_cube=0)
    values, analytic = collision.evaluate(q, state)
    eps = 1e-5
    numeric = np.stack(
        [
            (
                collision.evaluate(q + direction * eps, state)[0]
                - collision.evaluate(q - direction * eps, state)[0]
            )
            / (2 * eps)
            for direction in np.eye(6)
        ],
        axis=1,
    )
    # Cube 1 is an obstacle, regardless of the allowed grasp on cube 0.
    assert values[3] < 0 if offset[0] == 0.02 else values[3] > 0
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)
    automatic = jax.jit(jax.jacfwd(collision.distance))(jnp.array(q), jnp.array(state))
    np.testing.assert_allclose(automatic, analytic, atol=2e-6)


def test_carried_cube_moves_with_hand_and_keeps_obstacle_constraints(scene):
    model, data, collision, q = scene
    adr = model.joint("cube0_free").qposadr[0]
    data.qpos[adr : adr + 3] = data.site_xpos[collision.site]
    state = collision.context(data, allowed_cube=0, carried_cube=0)
    moved = q + np.array([0.15, 0.05, -0.04, 0.0, 0.0, 0.0])
    collision.evaluate(moved, state)
    np.testing.assert_allclose(
        collision.data.xpos[model.body("cube0").id],
        collision.data.site_xpos[collision.site],
        atol=1e-7,
    )
    # Put another cube directly on the moving payload, without moving the arm.
    overlap = collision.data.qpos[adr : adr + 3].copy()
    obstacle = model.joint("cube1_free").qposadr[0]
    state[obstacle : obstacle + 3] = overlap + [0.025, 0.0, 0.0]
    values, _ = collision.evaluate(moved, state)
    carried_cube1_group = len(collision.obstacles) + 2 + 3
    assert values[carried_cube1_group] < 0


def test_only_closed_fingers_are_exempt_for_the_active_cube(scene):
    model, data, collision, q = scene
    for side in ("left", "right"):
        data.qpos[model.joint(f"left_{side}_finger").qposadr] = 0.02
    mujoco.mj_forward(model, data)
    adr = model.joint("cube1_free").qposadr[0]
    data.qpos[adr : adr + 3] = data.site_xpos[collision.site]
    grasp = collision.context(data, allowed_cube=1)
    assert collision.evaluate(q, grasp)[0][3] > 0.03
    obstacle = collision.context(data)
    assert collision.evaluate(q, obstacle)[0][3] < 0.001
    for side in ("left", "right"):
        data.qpos[model.joint(f"left_{side}_finger").qposadr] = 0.0475
    mujoco.mj_forward(model, data)
    data.qpos[adr : adr + 3] = data.geom_xpos[model.geom("left_left_pad").id]
    open_hand = collision.context(data, allowed_cube=1)
    assert collision.evaluate(q, open_hand)[0][3] < -0.01


def test_optimizer_moves_out_of_an_obstacle_instead_of_only_rejecting(scene):
    import jax
    import jax.numpy as jnp
    import jaxls

    model, data, collision, q = scene
    adr = model.joint("cube1_free").qposadr[0]
    data.qpos[adr : adr + 3] = data.site_xpos[collision.site] + [0.02, 0.035, 0.005]
    state = jnp.asarray(collision.context(data, allowed_cube=0))
    assert collision.evaluate(q, state)[0][3] < -0.005

    class Config(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(6)):
        pass

    var = Config(0)

    @jaxls.Cost.factory()
    def stay_near(vals, var):
        return vals[var] - jnp.asarray(q)

    @jaxls.Cost.factory(kind="constraint_leq_zero")
    def avoid(vals, var):
        return 10 * (collision.margins - collision.distance(vals[var], state))

    problem = jaxls.LeastSquaresProblem([stay_near(var), avoid(var)], [var]).analyze()
    result = problem.solve(
        initial_vals=jaxls.VarValues.make([var.with_value(jnp.asarray(q))]),
        termination=jaxls.TerminationConfig(max_iterations=50),
        verbose=False,
    )
    final = collision.evaluate(np.asarray(result[var]), np.asarray(state))[0]
    assert final[3] > 0.002
    assert np.all(final >= -0.0002)
