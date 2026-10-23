"""Exercise actual jaxls and PyRoki APIs, not just standalone encoder calls."""

import io

import jax.numpy as jnp
import jaxls
import numpy as np
import pyroki as pk
import pytest
import yourdfpy

from datafarm_pyroki import JointMap
from datafarm_pyroki.integration import (
    InfeasibleTrajectoryError,
    make_motion_cost,
    solve_leg,
)


def robot_fixture(n_joints=7):
    xml = '<robot name="test_chain"><link name="base"/>'
    for i in range(1, n_joints + 1):
        parent = "base" if i == 1 else f"link{i - 1}"
        xml += f'''
        <link name="link{i}"><collision>
          <geometry><sphere radius="0.02"/></geometry>
        </collision></link>
        <joint name="joint{i}" type="revolute">
          <parent link="{parent}"/><child link="link{i}"/>
          <origin xyz="0 0 0.15"/><axis xyz="0 1 0"/>
          <limit lower="-3" upper="3" velocity="2" effort="100"/>
        </joint>'''
    xml += "</robot>"
    urdf = yourdfpy.URDF.load(io.BytesIO(xml.encode()), load_meshes=False)
    return pk.Robot.from_urdf(urdf), pk.collision.RobotCollision.from_urdf(urdf)


def test_one_leg_is_one_factor_with_correct_weight(encoder, motion):
    robot, _ = robot_fixture()
    mapping = JointMap.from_names(robot.joints.actuated_names, encoder.spec.joint_names)
    vs = tuple(robot.joint_var_cls(i) for i in range(len(motion)))
    cost = make_motion_cost(encoder, vs, source_dt=0.15, weight=3.0, joint_map=mapping)
    batch = robot.joint_var_cls(jnp.arange(len(motion)))
    values = jaxls.VarValues.make((batch.with_value(jnp.asarray(motion)),))
    problem = jaxls.LeastSquaresProblem(costs=[cost], variables=[batch]).analyze()
    residual = problem.compute_residual_vector(values)
    assert residual.shape == (16,)
    np.testing.assert_allclose(
        jnp.sum(residual**2), encoder.cost(jnp.asarray(motion), 0.15, 3), rtol=1e-5
    )
    with pytest.raises(ValueError, match="scalar-ID"):
        make_motion_cost(encoder, [batch, batch], source_dt=0.15, weight=1, joint_map=mapping)


def test_constrained_pyroki_solve(encoder):
    robot, collision = robot_fixture()
    mapping = JointMap.from_names(robot.joints.actuated_names, encoder.spec.joint_names)
    start = jnp.zeros(7)
    goal = jnp.array([0.04, -0.03, 0.02, -0.01, 0.03, -0.02, 0.01])
    obstacle = pk.collision.Sphere.from_center_and_radius(
        jnp.array([3.0, 3.0, 3.0]), jnp.array([0.1])
    )
    result = solve_leg(
        robot,
        encoder,
        start,
        goal,
        joint_map=mapping,
        timesteps=8,
        source_dt=0.15,
        encoder_weight=1e-6,
        acceleration_weight=0.001,
        acceleration_limits=jnp.full(7, 5.0),
        robot_coll=collision,
        world_coll=[obstacle],
        max_iterations=30,
    )
    np.testing.assert_allclose(result.positions[:2], jnp.stack([start, start]), atol=2e-3)
    np.testing.assert_allclose(result.positions[-2:], jnp.stack([goal, goal]), atol=2e-3)
    assert np.isfinite(result.final_style_cost)
    assert result.max_constraint_violation <= 2e-3
    assert np.max(np.abs(np.diff(result.positions, axis=0) / 0.15)) <= 2.002
    with pytest.raises(InfeasibleTrajectoryError, match="endpoint"):
        solve_leg(robot, encoder, start, jnp.full(7, 4.0), joint_map=mapping, source_dt=0.15)


def test_explicit_joint_mapping():
    mapping = JointMap.from_names(
        ["b", "aux", "a"], ["joint1", "joint2"], aliases={"joint1": "a", "joint2": "b"}
    )
    np.testing.assert_array_equal(mapping.select(jnp.array([1, 20, 3])), [3, 1])
    with pytest.raises(ValueError, match="missing"):
        JointMap.from_names(["a", "b"], ["joint1", "joint2"])
    with pytest.raises(ValueError, match="distinct"):
        JointMap.from_names(
            ["a", "b"], ["joint1", "joint2"], aliases={"joint1": "a", "joint2": "a"}
        )
