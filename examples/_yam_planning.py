"""Per-arm YAM planning with independent 6-DOF posture and motion priors."""

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
from _yam_scene import ARM_NAMES, HOME, STARTS, TARGETS, initial_state

from datafarm_pyroki import GripperEvent, JointMap, MotionLeg, load_encoder, load_joint_prior
from datafarm_pyroki.integration import make_ik_solver, make_leg_solver

N_WAYPOINTS = 24
SOURCE_DT = 0.1


def robot_contact(model, contact, allowed_cube=None):
    """Return whether a penetrating contact is unintended for this task."""
    names = [model.body(int(model.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2)]
    robot = [n.startswith(("left_", "right_")) for n in names]
    if not any(robot):
        return False
    if allowed_cube is not None and f"cube{allowed_cube}" in names:
        if any(("finger" in n or "pad" in n) for n in names):
            return False
    return True


class Planner:
    def __init__(self, model, urdf, arm, models, *, enabled, seed=7, weight=0.1, direct=False):
        self.model, self.arm, self.enabled = model, arm, enabled
        self.scratch = initial_state(model, settle=False)
        self.robot = pk.Robot.from_urdf(yourdfpy.URDF.load(urdf), default_joint_cfg=HOME)
        self.encoder = load_encoder(models / arm / "encoder.npz")
        self.prior = load_joint_prior(models / arm / "posture_prior.npz")
        if (
            self.encoder.spec.joint_names != ARM_NAMES[arm]
            or self.prior.joint_names != ARM_NAMES[arm]
        ):
            raise ValueError(f"Model joint names do not match {arm} YAM")
        if self.encoder.spec.sample_rate_hz != 30:
            raise ValueError("This demo expects the trained native 30 Hz YAM models")
        self.mapping = JointMap.from_names(
            self.robot.joints.actuated_names, self.encoder.spec.joint_names
        )
        self.tcp = self.robot.links.names.index("tcp")
        self.qpos = np.array([model.joint(n).qposadr[0] for n in ARM_NAMES[arm]])
        self.lower = np.asarray(self.robot.joints.lower_limits)
        self.upper = np.asarray(self.robot.joints.upper_limits)
        rng = np.random.default_rng(seed)
        self.seeds = rng.uniform(self.lower + 0.02, self.upper - 0.02, size=(64, 6))
        self.seeds[:24] = np.clip(
            HOME + rng.normal(size=(24, 6)) * 0.7, self.lower + 0.01, self.upper - 0.01
        )
        self.ik = make_ik_solver(self.robot, "tcp", seed_weight=0.001, max_iterations=100)
        self.fk = jax.jit(self.robot.forward_kinematics)
        self.ik_records, self.leg_records = [], []
        self.direct = direct
        self.carried_cube = None
        if direct:
            from _yam_collision import SceneDistances

            self.collisions = SceneDistances(model, arm)
        self._make_solvers(weight)
        for q in np.clip(HOME + rng.normal(size=(3, 6)) * 0.1, self.lower, self.upper):
            self.scratch.qpos[self.qpos] = q
            mujoco.mj_forward(model, self.scratch)
            np.testing.assert_allclose(
                np.asarray(self.fk(jnp.asarray(q)))[self.tcp, 4:],
                self.scratch.site_xpos[model.site(arm + "_grasp_site").id],
                atol=3e-6,
            )

    def _make_solvers(self, weight):
        robot, tcp = self.robot, self.tcp
        links = jnp.asarray([robot.links.names.index(f"{self.arm}_link_{i}") for i in range(2, 7)])
        sign = 1.0 if self.arm == "left" else -1.0

        @jaxls.Cost.factory(kind="constraint_leq_zero", name="yam_table_and_midline")
        def clearance(vals, var):
            poses = robot.forward_kinematics(vals[var])
            return jnp.concatenate((0.045 - poses[links, 6], 0.075 - sign * poses[links, 5]))

        @jaxls.Cost.factory(kind="constraint_leq_zero", name="yam_free_tcp_height")
        def free_height(vals, var):
            return 0.12 - robot.forward_kinematics(vals[var])[tcp, 6]

        def build(track):
            def extra(points, context):
                targets = context["targets"] if self.direct else context
                variables = robot.joint_var_cls(jnp.arange(len(points)))
                costs = [clearance(variables)]
                if track:
                    costs.append(
                        pk.costs.pose_cost(
                            jax.tree.map(lambda x: x[None], robot),
                            variables,
                            jaxlie.SE3(targets),
                            jnp.full(len(points), tcp, dtype=jnp.int32),
                            jnp.full(len(points), 250.0),
                            jnp.full(len(points), 80.0),
                        )
                    )
                elif not self.direct:
                    costs.append(free_height(variables))
                if self.direct:

                    @jaxls.Cost.factory(kind="constraint_leq_zero", name="scene_geometry")
                    def geometry(vals, a, b, u, state):
                        q = (1 - u) * vals[a] + u * vals[b]
                        return 10 * (self.collisions.margins - self.collisions.distance(q, state))

                    # Endpoints plus three interior samples of every interval;
                    # dense validation below checks the returned path again.
                    index = np.repeat(np.arange(len(points) - 1), 4)
                    blend = np.tile(np.array([0, 0.25, 0.5, 0.75]), len(points) - 1)
                    index = np.r_[index, len(points) - 2]
                    blend = np.r_[blend, 1.0]
                    costs.append(
                        geometry(
                            robot.joint_var_cls(jnp.asarray(index)),
                            robot.joint_var_cls(jnp.asarray(index + 1)),
                            jnp.asarray(blend[:, None]),
                            context["scene"][None],
                        )
                    )
                return costs

            return make_leg_solver(
                robot,
                self.encoder,
                joint_map=self.mapping,
                timesteps=N_WAYPOINTS,
                source_dt=SOURCE_DT,
                rest_endpoints=False,
                encoder_weight=weight if self.enabled else 0.0,
                smoothness_weight=0.25,
                acceleration_weight=0.0225,
                extra_costs=extra,
                max_iterations=320 if self.direct else 180,
            )

        self.optimizers = {track: build(track) for track in (False, True)}

    def valid(self, q, cube=None):
        if (
            not np.isfinite(q).all()
            or np.any(q < self.lower - 0.002)
            or np.any(q > self.upper + 0.002)
        ):
            return False
        self.scratch.qpos[self.qpos] = q
        mujoco.mj_forward(self.model, self.scratch)
        return not any(
            c.dist < -0.002 and robot_contact(self.model, c, cube) for c in self.scratch.contact
        )

    def endpoint(self, xyz, current, cube, *, orientation=None):
        # Ry(pi): downward tool axis, keeping wrist yaw inside the YAM's limits.
        if orientation is None:
            orientation = [0.0, 0.0, 1.0, 0.0]
        target = np.r_[orientation, xyz].astype(np.float32)
        seeds = self.seeds.copy()
        seeds[0] = current
        seeds[1] = HOME
        result = self.ik(target, seeds, feasibility=lambda q: self.valid(q, cube))
        if not result.feasible.any():
            raise RuntimeError(
                f"{self.arm}: no feasible IK at {xyz}; best error {result.position_errors.min()}"
            )
        q, valid = result.positions, result.feasible
        nll = np.asarray(self.prior.nll(jnp.asarray(q)))
        index = int(
            np.argmin(
                np.where(
                    valid, nll if self.enabled else np.linalg.norm(q - current, axis=1), np.inf
                )
            )
        )
        self.ik_records.append(
            dict(
                xyz=np.asarray(xyz).tolist(),
                feasible=int(valid.sum()),
                selected_nll=float(nll[index]),
            )
        )
        return q[index]

    def leg(self, start, goal, label, *, cube=None, contact=False):
        before = time.monotonic()
        u = np.linspace(0, 1, N_WAYPOINTS)
        initial = start + u[:, None] * (goal - start)
        endpoints = np.asarray(self.fk(jnp.asarray(np.stack([start, goal]))))[:, self.tcp]
        targets = np.empty((N_WAYPOINTS, 7), np.float32)
        a, b = (jaxlie.SO3(jnp.asarray(p[:4])) for p in endpoints)
        targets[:, :4] = np.asarray(
            (a @ jaxlie.SO3.exp(jnp.asarray(u[:, None]) * (a.inverse() @ b).log())).wxyz
        )
        targets[:, 4:] = endpoints[0, 4:] + u[:, None] * (endpoints[1, 4:] - endpoints[0, 4:])
        context = jnp.asarray(targets)
        if self.direct:
            self.scratch.qpos[self.qpos] = start
            state = self.collisions.context(
                self.scratch, allowed_cube=cube, carried_cube=self.carried_cube
            )
            self.last_collision_state = state
            context = dict(targets=jnp.asarray(targets), scene=jnp.asarray(state))
        q = np.asarray(
            self.optimizers[contact](
                start, goal, initial_waypoints=initial, context=context
            ).positions
        ).copy()
        poses = np.asarray(self.fk(jnp.asarray(q)))[:, self.tcp]
        error = float(np.linalg.norm(poses[:, 4:] - targets[:, 4:], axis=-1).max())
        grid = np.linspace(0, len(q) - 1, 5 * (len(q) - 1) + 1)
        dense = np.stack([np.interp(grid, np.arange(len(q)), q[:, j]) for j in range(6)], axis=1)
        if self.direct:
            distances = np.stack([self.collisions.evaluate(row, state)[0] for row in dense])
            collision_valid = bool(np.all(distances >= -0.0005))
        else:
            collision_valid = all(self.valid(row, cube) for row in dense)
        if (contact and error > 0.015) or not collision_valid:
            raise RuntimeError(
                f"{self.arm}/{label}: rejected collision or contact tracking ({error:.4g} m)"
            )
        record = dict(
            label=label,
            contact=contact,
            style_cost=float(self.encoder.cost(jnp.asarray(q), SOURCE_DT)),
            initial_style_cost=float(self.encoder.cost(jnp.asarray(initial), SOURCE_DT)),
            max_reference_deviation_m=error,
            planning_seconds=time.monotonic() - before,
        )
        if self.direct:
            record["minimum_geometry_distance_m"] = float(distances.min())
            record["carried_cube"] = self.carried_cube
        self.leg_records.append(record)
        print(
            f"[{self.arm}] {label}: style {record['style_cost']:.2f}, "
            f"deviation {error * 1000:.1f} mm",
            flush=True,
        )
        return q


def plan_arm(planner):
    arm = planner.arm
    current = HOME.copy()
    steps = []
    cubes = range(2) if arm == "left" else range(2, 4)
    for cube in cubes:

        def move(xyz, label, contact=False):
            nonlocal current
            goal = planner.endpoint(xyz, current, cube)
            label = f"cube {cube + 1}: {label}"
            steps.append(
                MotionLeg(
                    planner.leg(current, goal, label, cube=cube, contact=contact), SOURCE_DT, label
                )
            )
            current = goal

        start, target = STARTS[cube], TARGETS[cube]
        high_start = np.r_[start[:2], 0.14]
        high_target = np.r_[target[:2], 0.14]
        move(high_start, "approach")
        move(start, "descend", True)
        steps.append(
            GripperEvent(
                {"arm": arm, "action": "close", "cube": cube}, gripper=arm, dwell_seconds=0.65
            )
        )
        move(high_start, "lift", True)
        move(high_target, "transport")
        move(target, "lower", True)
        steps.append(
            GripperEvent(
                {"arm": arm, "action": "open", "cube": cube}, gripper=arm, dwell_seconds=0.55
            )
        )
        move(high_target, "retreat", True)
        address = planner.model.joint(f"cube{cube}_free").qposadr[0]
        planner.scratch.qpos[address : address + 3] = target
    steps.append(MotionLeg(planner.leg(current, HOME, "return home"), SOURCE_DT, "return home"))
    return steps
