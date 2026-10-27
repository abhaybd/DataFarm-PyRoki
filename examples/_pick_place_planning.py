"""PyRoki IK and whole-leg trajectory optimization for the contact manipulation demo."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import mujoco
import numpy as np
import pyroki as pk
import yourdfpy
from _pick_place_scene import HOME

from datafarm_pyroki import JointMap, load_encoder, load_joint_prior, select_ik_solution
from datafarm_pyroki.integration import make_motion_cost

N_WAYPOINTS = 20
SOURCE_DT = 0.10


class Planner:
    def __init__(self, model, urdf_path, *, datafarm, seed=7, encoder_weight=0.3):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.robot = pk.Robot.from_urdf(yourdfpy.URDF.load(urdf_path), default_joint_cfg=HOME)
        self.tcp = self.robot.links.names.index("tcp")
        self.mapping = JointMap.from_names(
            self.robot.joints.actuated_names, tuple(f"joint{i}" for i in range(1, 8))
        )
        self.encoder = load_encoder()
        self.prior = load_joint_prior()
        self.datafarm = datafarm
        self.lower = np.asarray(self.robot.joints.lower_limits)
        self.upper = np.asarray(self.robot.joints.upper_limits)
        self.qpos = np.array([model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.rng = np.random.default_rng(seed)
        self.seed_offsets = self.rng.normal(size=(24, 7)) * np.array(
            [0.8, 0.45, 0.8, 0.45, 0.8, 0.5, 1.0]
        )
        self.ik_records = []
        self.leg_records = []
        self._make_solvers(encoder_weight)
        self._check_kinematics()

    def _check_kinematics(self):
        for q in np.clip(HOME + self.seed_offsets[:5] * 0.2, self.lower, self.upper):
            self.scratch.qpos[self.qpos] = q
            mujoco.mj_forward(self.model, self.scratch)
            expected = self.scratch.site_xpos[self.model.site("rq_pinch").id]
            actual = np.asarray(self.fk(jnp.asarray(q)))[self.tcp, 4:]
            if np.max(np.abs(expected - actual)) > 3e-6:
                raise RuntimeError("PyRoki/MuJoCo TCP kinematics disagree")

    def _make_solvers(self, encoder_weight):
        robot, tcp, encoder, mapping = self.robot, self.tcp, self.encoder, self.mapping

        @jaxls.Cost.factory(name="seed_regularization")
        def seed_cost(vals, var, seed):
            return 0.02 * (vals[var] - seed)

        def one_ik(target, seed):
            var = robot.joint_var_cls(0)
            costs = [
                pk.costs.pose_cost(robot, var, jaxlie.SE3(target), jnp.int32(tcp), 150.0, 50.0),
                pk.costs.limit_constraint(robot, var),
                seed_cost(var, seed),
            ]
            solution = (
                jaxls.LeastSquaresProblem(costs, [var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make((var.with_value(seed),)),
                    termination=jaxls.TerminationConfig(max_iterations=80),
                    verbose=False,
                )
            )
            return solution[var]

        self.ik = jax.jit(jax.vmap(one_ik, in_axes=(None, 0)))

        @jaxls.Cost.factory(kind="constraint_eq_zero", name="endpoint")
        def endpoint(vals, var, target):
            return vals[var] - target

        @jaxls.Cost.factory(name="acceleration")
        def acceleration(vals, a, b, c):
            return 0.15 * (vals[c] - 2 * vals[b] + vals[a]) / SOURCE_DT**2

        @jaxls.Cost.factory(kind="constraint_leq_zero", name="arm_table_clearance")
        def table(vals, var):
            poses = robot.forward_kinematics(vals[var])
            indices = jnp.asarray([robot.links.names.index(f"link{i}") for i in range(2, 8)])
            return 0.075 - poses[indices, 6]

        @jaxls.Cost.factory(kind="constraint_leq_zero", name="free_space_table_clearance")
        def free_space_clearance(vals, var):
            # An inequality leaves the transit shape and orientation free while
            # keeping the gripper/carried cube above the tabletop.
            return 0.10 - robot.forward_kinematics(vals[var])[tcp, 6]

        def optimize(initial, targets, *, track_pose):
            n = N_WAYPOINTS
            variables = robot.joint_var_cls(jnp.arange(n))
            points = tuple(robot.joint_var_cls(i) for i in range(n))
            costs = [
                pk.costs.limit_constraint(jax.tree.map(lambda x: x[None], robot), variables),
                pk.costs.limit_velocity_constraint(
                    jax.tree.map(lambda x: x[None], robot),
                    robot.joint_var_cls(jnp.arange(1, n)),
                    robot.joint_var_cls(jnp.arange(n - 1)),
                    SOURCE_DT,
                ),
                pk.costs.smoothness_cost(
                    robot.joint_var_cls(jnp.arange(1, n)),
                    robot.joint_var_cls(jnp.arange(n - 1)),
                    0.5,
                ),
                acceleration(
                    robot.joint_var_cls(jnp.arange(n - 2)),
                    robot.joint_var_cls(jnp.arange(1, n - 1)),
                    robot.joint_var_cls(jnp.arange(2, n)),
                ),
                table(variables),
                endpoint(points[0], initial[0]),
                endpoint(points[-1], initial[-1]),
            ]
            if track_pose:
                costs.append(
                    pk.costs.pose_cost(
                        jax.tree.map(lambda x: x[None], robot),
                        variables,
                        jaxlie.SE3(targets),
                        jnp.full(n, tcp, dtype=jnp.int32),
                        jnp.full(n, 250.0),
                        jnp.full(n, 80.0),
                    )
                )
            else:
                costs.append(free_space_clearance(variables))
            if self.datafarm:
                costs.append(
                    make_motion_cost(
                        encoder,
                        points,
                        source_dt=SOURCE_DT,
                        weight=encoder_weight,
                        joint_map=mapping,
                    )
                )
            solution = (
                jaxls.LeastSquaresProblem(costs, [variables])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make((variables.with_value(initial),)),
                    termination=jaxls.TerminationConfig(max_iterations=65),
                    verbose=False,
                )
            )
            return solution[variables]

        self.optimize = jax.jit(optimize, static_argnames=("track_pose",))
        self.fk = jax.jit(robot.forward_kinematics)

    def valid_configuration(self, q, allowed_cube=None):
        if (
            not np.isfinite(q).all()
            or np.any(q < self.lower - 0.002)
            or np.any(q > self.upper + 0.002)
        ):
            return False
        self.scratch.qpos[self.qpos] = q
        mujoco.mj_forward(self.model, self.scratch)
        for contact in self.scratch.contact:
            if contact.dist >= -0.002:
                continue
            b1, b2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            names = [self.model.body(int(b)).name for b in (b1, b2)]
            robot_flags = [name.startswith(("link", "rq_", "attachment")) for name in names]
            if not any(robot_flags):
                continue
            if all(name.startswith("rq_") for name in names):
                continue
            if "link0" in names and "world" in names:
                continue
            geom_names = [self.model.geom(int(g)).name for g in (contact.geom1, contact.geom2)]
            if (
                allowed_cube is not None
                and f"cube{allowed_cube}" in names
                and any(name.startswith("rq_") for name in names)
            ):
                continue
            # Resting cube/table contacts are unrelated to robot feasibility.
            if (
                all(robot_flags)
                or "table" in geom_names
                or any(name.startswith("cube") for name in names)
            ):
                return False
        return True

    def endpoint(self, xyz, current, *, allowed_cube=None):
        target = np.r_[[0.0, 1.0, 0.0, 0.0], xyz].astype(np.float32)
        seeds = np.clip(HOME + self.seed_offsets, self.lower + 0.02, self.upper - 0.02)
        seeds[0] = current
        seeds[1] = HOME
        candidates = np.asarray(self.ik(jnp.asarray(target), jnp.asarray(seeds, dtype=jnp.float32)))
        poses = np.asarray(self.fk(jnp.asarray(candidates)))[:, self.tcp]
        position_error = np.linalg.norm(poses[:, 4:] - xyz, axis=-1)
        orientation_error = 2 * np.arccos(np.clip(np.abs(poses[:, :4] @ target[:4]), 0, 1))
        feasible = (position_error < 0.0015) & (orientation_error < 0.015)
        feasible &= np.array([self.valid_configuration(q, allowed_cube) for q in candidates])
        if not feasible.any():
            raise RuntimeError(
                f"No feasible IK for {xyz}; best position error {position_error.min():.4g}"
            )
        nll = np.asarray(self.prior.nll(jnp.asarray(candidates)))
        if self.datafarm:
            result = select_ik_solution(
                self.prior,
                candidates,
                joint_map=self.mapping,
                feasible=feasible,
                lower_limits=self.lower,
                upper_limits=self.upper,
            )
            index = int(result.index)
        else:
            index = int(
                np.argmin(np.where(feasible, np.linalg.norm(candidates - current, axis=1), np.inf))
            )
        self.ik_records.append(
            {
                "xyz": np.asarray(xyz).tolist(),
                "feasible_candidates": int(feasible.sum()),
                "selected": index,
                "selected_nll": float(nll[index]),
                "best_feasible_nll": float(nll[feasible].min()),
            }
        )
        return candidates[index]

    def leg(self, start, goal, label, *, allowed_cube=None, track_pose=False):
        """Fix joint endpoints; only contact legs track intermediate TCP poses."""
        before = time.monotonic()
        u = np.linspace(0.0, 1.0, N_WAYPOINTS)
        initial = start + u[:, None] * (goal - start)
        endpoints = np.asarray(self.fk(jnp.asarray(np.stack([start, goal]))))[:, self.tcp]
        targets = np.empty((N_WAYPOINTS, 7), np.float32)
        targets[:, :4] = endpoints[1, :4]
        # The initial home move blends its orientation as well as position.
        a, b = jaxlie.SO3(jnp.asarray(endpoints[0, :4])), jaxlie.SO3(jnp.asarray(endpoints[1, :4]))
        targets[:, :4] = np.asarray(
            (a @ jaxlie.SO3.exp(jnp.asarray(u[:, None]) * (a.inverse() @ b).log())).wxyz
        )
        targets[:, 4:] = endpoints[0, 4:] + u[:, None] * (endpoints[1, 4:] - endpoints[0, 4:])
        q = np.asarray(
            self.optimize(
                jnp.asarray(initial, dtype=jnp.float32), jnp.asarray(targets), track_pose=track_pose
            )
        ).copy()
        q[0], q[-1] = start, goal
        poses = np.asarray(self.fk(jnp.asarray(q)))[:, self.tcp]
        max_error = float(np.linalg.norm(poses[:, 4:] - targets[:, 4:], axis=-1).max())
        dense_grid = np.linspace(0, len(q) - 1, 5 * (len(q) - 1) + 1)
        dense = np.stack([np.interp(dense_grid, np.arange(len(q)), q[:, j]) for j in range(7)], -1)
        velocity = np.diff(q, axis=0) / SOURCE_DT
        valid_limits = np.all(
            np.abs(velocity) <= np.asarray(self.robot.joints.velocity_limits) + 0.005
        )
        if (
            (track_pose and max_error > 0.015)
            or (
                not track_pose
                and np.min(np.asarray(self.fk(jnp.asarray(dense)))[:, self.tcp, 6]) < 0.098
            )
            or not valid_limits
            or not all(self.valid_configuration(row, allowed_cube) for row in dense)
        ):
            raise RuntimeError(
                f"Rejected {label}: tracking error {max_error:.4g}, motion limit or collision"
            )
        record = {
            "label": label,
            "motion_type": "contact" if track_pose else "free",
            "fixed_joint_waypoints": [0, N_WAYPOINTS - 1],
            "style_cost": float(self.encoder.cost(jnp.asarray(q), SOURCE_DT)),
            "max_tracking_error_m": max_error if track_pose else None,
            "max_reference_deviation_m": max_error,
            "planning_seconds": time.monotonic() - before,
        }
        self.leg_records.append(record)
        deviation_label = "tracking error" if track_pose else "free path deviation"
        print(
            f"  {label}: style {record['style_cost']:.3f}, "
            f"{deviation_label} {max_error * 1000:.1f} mm",
            flush=True,
        )
        return q
