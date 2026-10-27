"""Optional PyRoki/jaxls integration. Install with `uv sync --extra planning`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import jaxls
import numpy as np
import pyroki as pk

from .embodiment import JointMap
from .encoder import MotionEncoder, sample_count


def make_motion_cost(
    encoder: MotionEncoder,
    joint_vars: Sequence[jaxls.Var],
    *,
    source_dt: float,
    weight: float,
    joint_map: JointMap,
) -> jaxls.Cost:
    """One 16-D (or model-specific D-D) residual spanning one complete motion leg.

    Pass a tuple/list of scalar-ID timestep variables, NOT a batched Var as the
    sole argument: jaxls would otherwise vmap the encoder over individual timesteps.
    Reverse-mode Jacobians suit a small latent and many trajectory coordinates.
    The encoder is frozen in this closure; only joint_vars are optimization variables.
    """
    variables = tuple(joint_vars)
    sample_count(len(variables), source_dt, encoder.spec.sample_rate_hz)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be nonnegative and finite")
    if joint_map.model_joint_names != encoder.spec.joint_names:
        raise ValueError("JointMap does not match this model's ordered joint names")
    if any(np.shape(var.id) != () for var in variables):
        raise ValueError("Pass individual scalar-ID variables, one per waypoint")
    if len({(type(v), int(v.id)) for v in variables}) != len(variables):
        raise ValueError("Motion leg contains duplicate timestep variables")

    @jaxls.Cost.factory(jac_mode="reverse", name=f"motion_style:{encoder.spec.name}")
    def motion_style(vals: jaxls.VarValues, waypoints: tuple[jaxls.Var, ...]) -> jax.Array:
        q = jnp.stack([vals[var] for var in waypoints])
        return encoder.residual(joint_map.select(q), source_dt, weight).reshape(-1)

    return motion_style(variables)


@dataclass(frozen=True)
class LegResult:
    """Host-side result of a constrained solve, with model scores for inspection."""

    positions: jax.Array
    source_dt: float
    initial_style_cost: float
    final_style_cost: float
    max_constraint_violation: float


class InfeasibleTrajectoryError(RuntimeError):
    """The local optimizer did not return a trajectory satisfying the requested constraints."""


def solve_leg(
    robot: pk.Robot,
    encoder: MotionEncoder,
    start_cfg: jax.Array,
    goal_cfg: jax.Array,
    *,
    joint_map: JointMap,
    source_dt: float,
    initial_waypoints: jax.Array | None = None,
    timesteps: int = 32,
    encoder_weight: float = 1.0,
    smoothness_weight: float = 1.0,
    acceleration_weight: float = 0.1,
    acceleration_limits: jax.Array | None = None,
    robot_coll: pk.collision.RobotCollision | None = None,
    world_coll: Sequence[pk.collision.CollGeom] = (),
    collision_margin: float = 0.02,
    rest_endpoints: bool = True,
    extra_costs: Callable[[tuple[jaxls.Var, ...]], Sequence[jaxls.Cost]] | None = None,
    max_iterations: int = 50,
    constraint_tolerance: float = 2e-3,
    verbose: bool = False,
) -> LegResult:
    """Refine one motion leg with fixed joint endpoints and a frozen motion-style cost.

    This replaces the waypoint-optimization stage, not task planning, IK, gripper
    scheduling, or trajectory segment retiming. Supply the actual start and an already selected
    goal (e.g. from your endpoint prior). The uniform waypoint clock stays fixed.

    Self collisions are constrained at waypoints; world collisions use PyRoki's
    swept-capsule approximation. Add task/attached-object constraints through
    extra_costs, and perform application-specific continuous feasibility checks.
    No robot execution occurs here. A rejected solve raises, never returns success.
    """
    robot_names = tuple(robot.joints.actuated_names)
    dof = len(robot_names)
    if robot_names != joint_map.robot_joint_names:
        raise ValueError("JointMap must use the robot's actuated joint order")
    start = jnp.asarray(start_cfg, dtype=jnp.float32)
    goal = jnp.asarray(goal_cfg, dtype=jnp.float32)
    if start.shape != (dof,) or goal.shape != (dof,):
        raise ValueError(f"start_cfg and goal_cfg must have shape ({dof},)")
    if initial_waypoints is None:
        initial = jnp.linspace(start, goal, timesteps)
    else:
        initial = jnp.asarray(initial_waypoints, dtype=jnp.float32)
        if initial.ndim != 2 or initial.shape[1] != dof:
            raise ValueError("initial_waypoints must have shape [time,robot_dof]")
        timesteps = initial.shape[0]
    sample_count(timesteps, source_dt, encoder.spec.sample_rate_hz)
    if timesteps < 3:
        raise ValueError("The leg solver needs at least three waypoints")
    if rest_endpoints and timesteps < 4:
        raise ValueError("rest_endpoints requires at least four waypoints")
    if (
        not np.isfinite(initial).all()
        or not np.isfinite(start).all()
        or not np.isfinite(goal).all()
    ):
        raise ValueError("Configurations must be finite")
    if world_coll and robot_coll is None:
        raise ValueError("World collisions require a RobotCollision model")
    if any(
        not np.isfinite(w) or w < 0
        for w in (encoder_weight, smoothness_weight, acceleration_weight, collision_margin)
    ):
        raise ValueError("Weights and collision margin must be nonnegative and finite")
    if constraint_tolerance <= 0 or max_iterations <= 0:
        raise ValueError("constraint_tolerance and max_iterations must be positive")
    if acceleration_limits is not None:
        acceleration_limits = jnp.asarray(acceleration_limits)
        if (
            acceleration_limits.shape != (dof,)
            or not np.isfinite(acceleration_limits).all()
            or np.any(acceleration_limits <= 0)
        ):
            raise ValueError("acceleration_limits must contain one positive finite limit per joint")

    # Reject impossible endpoints before paying for an encoder Jacobian compilation.
    for endpoint in (start, goal):
        full = np.asarray(robot.joints.get_full_config(endpoint))
        if np.any(
            full < np.asarray(robot.joints.lower_limits_all) - constraint_tolerance
        ) or np.any(full > np.asarray(robot.joints.upper_limits_all) + constraint_tolerance):
            raise InfeasibleTrajectoryError("An endpoint violates joint limits")

    initial = initial.at[0].set(start).at[-1].set(goal)
    if rest_endpoints:
        initial = initial.at[1].set(start).at[-2].set(goal)
    batched_vars = robot.joint_var_cls(jnp.arange(timesteps))
    waypoints = tuple(robot.joint_var_cls(i) for i in range(timesteps))
    costs = [
        make_motion_cost(
            encoder, waypoints, source_dt=source_dt, weight=encoder_weight, joint_map=joint_map
        ),
        pk.costs.limit_constraint(jax.tree.map(lambda x: x[None], robot), batched_vars),
        pk.costs.limit_velocity_constraint(
            jax.tree.map(lambda x: x[None], robot),
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(timesteps - 1)),
            source_dt,
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(timesteps - 1)),
            jnp.sqrt(smoothness_weight),
        ),
    ]

    @jaxls.Cost.factory(kind="constraint_eq_zero", name="fixed_endpoint")
    def fixed_endpoint(vals, var, target):
        return vals[var] - target

    for index in (0, 1) if rest_endpoints else (0,):
        costs.append(fixed_endpoint(waypoints[index], start))
    for index in (timesteps - 2, timesteps - 1) if rest_endpoints else (timesteps - 1,):
        costs.append(fixed_endpoint(waypoints[index], goal))

    def acceleration(vals, a, b, c):
        return (vals[c] - 2 * vals[b] + vals[a]) / source_dt**2

    acc_args = (
        robot.joint_var_cls(jnp.arange(timesteps - 2)),
        robot.joint_var_cls(jnp.arange(1, timesteps - 1)),
        robot.joint_var_cls(jnp.arange(2, timesteps)),
    )
    costs.append(_acceleration_cost(acceleration, acc_args, acceleration_weight))
    if acceleration_limits is not None:

        @jaxls.Cost.factory(kind="constraint_leq_zero", name="acceleration_limit")
        def acceleration_limit(vals, a, b, c):
            return jnp.abs(acceleration(vals, a, b, c)) - acceleration_limits

        costs.append(acceleration_limit(*acc_args))

    if robot_coll is not None:

        @jaxls.Cost.factory(kind="constraint_leq_zero", name="self_collision")
        def self_collision(vals, var):
            return (
                collision_margin - robot_coll.compute_self_collision_distance(robot, vals[var])
            ).reshape(-1)

        costs.append(self_collision(batched_vars))

        @jaxls.Cost.factory(kind="constraint_leq_zero", name="world_collision_sweep")
        def world_collision(vals, previous, current, obstacle):
            capsules = robot_coll.get_swept_capsules(robot, vals[previous], vals[current])
            distance = pk.collision.collide(capsules.reshape((-1, 1)), obstacle.reshape((1, -1)))
            return (collision_margin - distance).reshape(-1)

        for obstacle in world_coll:
            costs.append(
                world_collision(
                    robot.joint_var_cls(jnp.arange(timesteps - 1)),
                    robot.joint_var_cls(jnp.arange(1, timesteps)),
                    jax.tree.map(lambda x: x[None], obstacle),
                )
            )

    if extra_costs is not None:
        costs.extend(extra_costs(waypoints))
    problem = jaxls.LeastSquaresProblem(costs=costs, variables=[batched_vars]).analyze()
    solution = problem.solve(
        initial_vals=jaxls.VarValues.make((batched_vars.with_value(initial),)),
        termination=jaxls.TerminationConfig(max_iterations=max_iterations),
        verbose=verbose,
    )
    # Check each constraint in its original convention: equality |r|, inequality signed.
    violation = 0.0
    for cost in costs:
        if cost.kind == "l2_squared":
            continue
        residual = np.asarray(_evaluate_cost(cost, solution))
        if not np.isfinite(residual).all():
            raise InfeasibleTrajectoryError("Nonfinite constraint residual after optimization")
        if cost.kind == "constraint_eq_zero":
            residual = np.abs(residual)
        elif cost.kind == "constraint_geq_zero":
            residual = -residual
        violation = max(violation, float(np.max(residual, initial=0)))
    result = solution[batched_vars]
    if not np.isfinite(result).all() or violation > constraint_tolerance:
        raise InfeasibleTrajectoryError(
            f"Constraint violation {violation:.6g} exceeds tolerance {constraint_tolerance}"
        )
    return LegResult(
        result,
        source_dt,
        float(encoder.cost(joint_map.select(initial), source_dt)),
        float(encoder.cost(joint_map.select(result), source_dt)),
        violation,
    )


def _acceleration_cost(residual_fn, args, weight):
    @jaxls.Cost.factory(name="acceleration")
    def weighted(vals, a, b, c):
        return jnp.sqrt(weight) * residual_fn(vals, a, b, c)

    return weighted(*args)


def _evaluate_cost(cost: jaxls.Cost, values: jaxls.VarValues) -> jax.Array:
    """Evaluate a cost using the same leading-axis batching convention as jaxls."""
    leaves = jax.tree.leaves(cost.args, is_leaf=lambda x: isinstance(x, jaxls.Var))
    variables = [x for x in leaves if isinstance(x, jaxls.Var)]
    batch_shape = np.broadcast_shapes(*(np.shape(var.id) for var in variables))
    if not batch_shape:
        result = cost.compute_residual(values, *cost.args)
    else:
        rank = len(batch_shape)
        args = jax.tree.map(
            lambda x: jnp.broadcast_to(jnp.asarray(x), batch_shape + np.shape(x)[rank:]),
            cost.args,
        )

        def evaluate(args):
            return cost.compute_residual(values, *args)

        for _ in batch_shape:
            evaluate = jax.vmap(evaluate)
        result = evaluate(args)
    # Custom-Jacobian factories can optionally return a cache with the residual.
    return result[0] if isinstance(result, tuple) else result
