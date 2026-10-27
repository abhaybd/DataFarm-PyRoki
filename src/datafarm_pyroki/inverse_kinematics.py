"""Optional PyRoki IK candidate generation, independent of any simulator or posture prior."""

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk


class IKCandidates(NamedTuple):
    positions: np.ndarray
    feasible: np.ndarray
    position_errors: np.ndarray
    orientation_errors: np.ndarray


@dataclass(frozen=True)
class IKSolver:
    """A compiled multi-seed IK solve with task-specific feasibility supplied by the caller."""

    _solve: Callable
    robot: pk.Robot
    link_index: int
    position_tolerance: float
    orientation_tolerance: float
    joint_tolerance: float

    def __call__(self, target, seeds, *, feasibility=None) -> IKCandidates:
        """Target is SE3 [w,x,y,z,px,py,pz]; seeds use actuated robot joint order.

        ``feasibility(q)`` may check collisions or other application constraints.
        A low position/orientation error alone never implies collision feasibility.
        Candidate order is retained so callers can rank using their own objective.
        """
        target = np.asarray(target, dtype=np.float32)
        seeds = np.asarray(seeds, dtype=np.float32)
        dof = len(self.robot.joints.actuated_names)
        if (
            target.shape != (7,)
            or not np.isfinite(target).all()
            or not np.isclose(np.linalg.norm(target[:4]), 1.0, atol=1e-5)
        ):
            raise ValueError("target must be a finite SE3 with a unit wxyz quaternion")
        if (
            seeds.ndim != 2
            or seeds.shape[0] == 0
            or seeds.shape[1] != dof
            or not np.isfinite(seeds).all()
        ):
            raise ValueError(f"seeds must be finite [candidates>=1,{dof}]")
        positions, poses, full = self._solve(jnp.asarray(target), jnp.asarray(seeds))
        positions, poses, full = map(np.asarray, (positions, poses, full))
        position_error = np.linalg.norm(poses[:, 4:] - target[4:], axis=-1)
        orientation_error = 2 * np.arccos(np.clip(np.abs(poses[:, :4] @ target[:4]), 0, 1))
        feasible = (
            np.isfinite(positions).all(-1)
            & (position_error < self.position_tolerance)
            & (orientation_error < self.orientation_tolerance)
            & np.all(
                full >= np.asarray(self.robot.joints.lower_limits_all) - self.joint_tolerance,
                axis=-1,
            )
            & np.all(
                full <= np.asarray(self.robot.joints.upper_limits_all) + self.joint_tolerance,
                axis=-1,
            )
        )
        if feasibility is not None:
            for i in np.flatnonzero(feasible):
                feasible[i] = bool(feasibility(positions[i]))
        return IKCandidates(positions, feasible, position_error, orientation_error)


def make_ik_solver(
    robot: pk.Robot,
    link_name: str,
    *,
    position_weight: float = 150.0,
    orientation_weight: float = 50.0,
    seed_weight: float = 0.02,
    max_iterations: int = 80,
    position_tolerance: float = 0.0015,
    orientation_tolerance: float = 0.015,
    joint_tolerance: float = 0.002,
) -> IKSolver:
    """Create once and reuse across target poses with explicitly supplied seeds.

    Weights scale residuals, matching PyRoki's pose_cost convention. Configuration
    sampling and endpoint ranking remain independent of candidate generation.
    """
    if link_name not in robot.links.names:
        raise ValueError(f"Unknown robot link {link_name!r}")
    if any(
        not np.isfinite(x) or x < 0
        for x in (position_weight, orientation_weight, seed_weight, joint_tolerance)
    ):
        raise ValueError("IK weights and joint_tolerance must be finite and nonnegative")
    if (
        any(not np.isfinite(x) or x <= 0 for x in (position_tolerance, orientation_tolerance))
        or max_iterations <= 0
    ):
        raise ValueError("IK tolerances and max_iterations must be positive")
    link = robot.links.names.index(link_name)

    @jaxls.Cost.factory(name="seed_regularization")
    def seed_cost(vals, var, seed):
        return seed_weight * (vals[var] - seed)

    def one_ik(target, seed):
        var = robot.joint_var_cls(0)
        costs = [
            pk.costs.pose_cost(
                robot, var, jaxlie.SE3(target), jnp.int32(link), position_weight, orientation_weight
            ),
            pk.costs.limit_constraint(robot, var),
            seed_cost(var, seed),
        ]
        solution = (
            jaxls.LeastSquaresProblem(costs, [var])
            .analyze()
            .solve(
                initial_vals=jaxls.VarValues.make((var.with_value(seed),)),
                termination=jaxls.TerminationConfig(max_iterations=max_iterations),
                verbose=False,
            )
        )
        return solution[var]

    @jax.jit
    def solve(target, seeds):
        q = jax.vmap(one_ik, in_axes=(None, 0))(target, seeds)
        return q, robot.forward_kinematics(q)[:, link], jax.vmap(robot.joints.get_full_config)(q)

    return IKSolver(solve, robot, link, position_tolerance, orientation_tolerance, joint_tolerance)
