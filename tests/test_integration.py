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
    np.testing.assert_array_equal(np.asarray(result.positions)[-1], np.asarray(goal))
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


def test_reusable_leg_solver_updates_task_context_without_retracing(encoder):
    from datafarm_pyroki.integration import make_leg_solver

    robot, _ = robot_fixture()
    mapping = JointMap.from_names(robot.joints.actuated_names, encoder.spec.joint_names)
    traces = []

    @jaxls.Cost.factory(kind="constraint_eq_zero", name="middle_target")
    def middle_target(vals, var, target):
        return vals[var] - target

    def task_costs(points, target):
        traces.append(True)
        return [middle_target(points[3], target)]

    solver = make_leg_solver(
        robot,
        encoder,
        joint_map=mapping,
        source_dt=0.15,
        timesteps=8,
        rest_endpoints=False,
        encoder_weight=0,
        smoothness_weight=0.01,
        acceleration_weight=0.0001,
        extra_costs=task_costs,
        max_iterations=40,
    )
    start = jnp.zeros(7)
    for displacement in (0.01, 0.02):
        target = jnp.full(7, displacement)
        result = solver(start, start, context=target)
        np.testing.assert_array_equal(np.asarray(result.positions)[[0, -1]], np.zeros((2, 7)))
        np.testing.assert_allclose(result.positions[3], target, atol=2e-3)
        assert result.max_constraint_violation <= 2e-3
    assert len(traces) == 1

    # The convenient single-leg API keeps its existing one-argument callback.
    result = solve_leg(
        robot,
        encoder,
        start,
        start,
        joint_map=mapping,
        source_dt=0.15,
        timesteps=8,
        rest_endpoints=False,
        encoder_weight=0,
        extra_costs=lambda points: [middle_target(points[3], jnp.full(7, 0.01))],
    )
    np.testing.assert_allclose(result.positions[3], 0.01, atol=2e-3)


def test_ik_generation_supports_other_robots_and_explicit_feasibility():
    from datafarm_pyroki.integration import make_ik_solver

    robot, _ = robot_fixture(3)
    goal = jnp.array([0.1, -0.2, 0.1])
    target = np.asarray(robot.forward_kinematics(goal))[-1]
    seeds = np.stack([goal, goal * 0.9, -goal])
    solver = make_ik_solver(robot, "link3")
    candidates = solver(target, seeds)
    assert candidates.positions.shape == seeds.shape
    assert candidates.feasible[0]
    assert candidates.position_errors[0] < 0.0015
    assert candidates.orientation_errors[0] < 0.015
    rejected = solver(target, seeds, feasibility=lambda q: False)
    assert not rejected.feasible.any()
    np.testing.assert_allclose(rejected.positions, candidates.positions)
    unreachable = target.copy()
    unreachable[4:] = [100, 100, 100]
    assert not solver(unreachable, seeds).feasible.any()
    with pytest.raises(ValueError, match="seeds"):
        solver(target, np.zeros((2, 7)))
    with pytest.raises(ValueError, match="quaternion"):
        solver(np.zeros(7), seeds)
    with pytest.raises(ValueError, match="Unknown"):
        make_ik_solver(robot, "missing")
