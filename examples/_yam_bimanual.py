"""Joint YAM path optimization with either independent or bimanual motion costs."""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import mujoco
import numpy as np
import pyroki as pk
import yourdfpy
from _yam_collision import SceneDistances
from _yam_planning import N_WAYPOINTS, SOURCE_DT
from _yam_scene import ARMS, HOME, JOINT_NAMES, initial_state

from datafarm_pyroki import EncoderSpec, JointMap, load_encoder
from datafarm_pyroki.encoder import MotionEncoder
from datafarm_pyroki.integration import InfeasibleTrajectoryError, make_leg_solver


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IndependentMotionEncoder:
    """Average the marginal costs on a shared path/clock, with no cross-arm latent.

    The precision blocks are divided by sqrt(2), so the style term is the mean
    of the two 16-dimensional Mahalanobis costs, matching the joint model's
    nominal 16-dimensional reference cost scale. This is an objective adapter,
    not a learned joint distribution.
    """

    left: MotionEncoder
    right: MotionEncoder

    def __post_init__(self):
        if self.left.spec.sample_rate_hz != self.right.spec.sample_rate_hz:
            raise ValueError("Independent encoder sample rates must match")

    @property
    def spec(self):
        return EncoderSpec(
            "yam-independent-average",
            self.left.spec.joint_names + self.right.spec.joint_names,
            self.left.spec.sample_rate_hz,
        )

    @property
    def latent_mean(self):
        return jnp.concatenate((self.left.latent_mean, self.right.latent_mean))

    @property
    def precision_cholesky(self):
        a, b = self.left.precision_cholesky, self.right.precision_cholesky
        return jnp.block(
            [[a, jnp.zeros((len(a), len(b)))], [jnp.zeros((len(b), len(a))), b]]
        ) / jnp.sqrt(2.0)

    def features_at_model_rate(self, q):
        n = self.left.spec.n_joints
        return jnp.concatenate(
            (
                self.left.features_at_model_rate(q[..., :n]),
                self.right.features_at_model_rate(q[..., n:]),
            ),
            axis=-2,
        )

    def encode(self, x, mask=None, *, padding_exact=False):
        channels = 4 * self.left.spec.n_joints
        a = self.left.encode(x[..., :channels, :], mask, padding_exact=padding_exact)
        b = self.right.encode(x[..., channels:, :], mask, padding_exact=padding_exact)
        return tuple(jnp.concatenate((a[i], b[i]), axis=-1) for i in (0, 1))

    def residual(self, q, source_dt, weight=1.0):
        n = self.left.spec.n_joints
        return jnp.concatenate(
            (
                self.left.residual(q[..., :n], source_dt, weight / 2),
                self.right.residual(q[..., n:], source_dt, weight / 2),
            ),
            axis=-1,
        )

    def cost(self, q, source_dt, weight=1.0):
        return jnp.sum(self.residual(q, source_dt, weight) ** 2, axis=-1)


class BimanualPlanner:
    """One 12-joint solve and clock per task phase, common to all three variants."""

    def __init__(self, model, urdf, models, variant, *, weight=0.1):
        self.model, self.variant = model, variant
        self.enabled = variant != "off"
        if variant == "bimanual":
            self.encoder = load_encoder(models / "bimanual/encoder.npz")
        else:
            self.encoder = IndependentMotionEncoder(
                load_encoder(models / "left/encoder.npz"),
                load_encoder(models / "right/encoder.npz"),
            )
        if self.encoder.spec.joint_names != JOINT_NAMES or self.encoder.spec.sample_rate_hz != 30:
            raise ValueError("Expected native 30 Hz left_joint1..6 then right_joint1..6")
        self.robot = pk.Robot.from_urdf(
            yourdfpy.URDF.load(urdf), default_joint_cfg=np.tile(HOME, 2)
        )
        if tuple(self.robot.joints.actuated_names) != JOINT_NAMES:
            raise ValueError("Combined URDF must preserve canonical joint order")
        self.mapping = JointMap.from_names(JOINT_NAMES, JOINT_NAMES)
        self.tcp = [self.robot.links.names.index(f"{arm}_tcp") for arm in ARMS]
        self.fk = jax.jit(self.robot.forward_kinematics)
        self.geometry = [SceneDistances(model, arm, coupled=True) for arm in ARMS]
        self.records = []
        self.source_paths = []
        self._make_solver(weight)
        self._check_fk()

    def _check_fk(self):
        data = initial_state(self.model, settle=False)
        indices = [self.model.joint(n).qposadr[0] for n in JOINT_NAMES]
        for q in np.tile(HOME, 2) + np.random.default_rng(9).normal(size=(3, 12)) * 0.08:
            data.qpos[indices] = q
            mujoco.mj_kinematics(self.model, data)
            poses = np.asarray(self.fk(jnp.asarray(q)))
            for arm, tcp in zip(ARMS, self.tcp, strict=True):
                np.testing.assert_allclose(
                    poses[tcp, 4:],
                    data.site_xpos[self.model.site(f"{arm}_grasp_site").id],
                    atol=3e-6,
                )

    def _make_solver(self, weight):
        robot = self.robot
        link_ids = jnp.asarray(
            [robot.links.names.index(f"{arm}_link_{i}") for arm in ARMS for i in range(2, 7)]
        )
        signs = jnp.repeat(jnp.array([1.0, -1.0]), 5)

        @jaxls.Cost.factory(kind="constraint_leq_zero", name="bimanual_table_midline")
        def clearance(vals, var):
            poses = robot.forward_kinematics(vals[var])
            return jnp.concatenate((0.045 - poses[link_ids, 6], 0.075 - signs * poses[link_ids, 5]))

        @jaxls.Cost.factory(kind="constraint_eq_zero", name="stationary_arm")
        def stationary(vals, var, start, moving):
            return 10 * (vals[var] - start) * (1 - moving)

        @jaxls.Cost.factory(kind="constraint_leq_zero", name="bimanual_scene_geometry")
        def geometry(vals, a, b, u, left_state, right_state):
            q = (1 - u) * vals[a] + u * vals[b]
            return jnp.concatenate(
                tuple(
                    10 * (g.margins - g.distance(q, state))
                    for g, state in zip(self.geometry, (left_state, right_state), strict=True)
                )
            )

        def extra(points, ctx):
            variables = robot.joint_var_cls(jnp.arange(len(points)))
            costs = [
                clearance(variables),
                stationary(variables, ctx["start"][None], ctx["moving"][None]),
            ]
            for i, tcp in enumerate(self.tcp):
                costs.append(
                    pk.costs.pose_cost(
                        jax.tree.map(lambda x: x[None], robot),
                        variables,
                        jaxlie.SE3(ctx["targets"][i]),
                        jnp.full(len(points), tcp, dtype=jnp.int32),
                        jnp.full(len(points), 250.0) * ctx["track"][i],
                        jnp.full(len(points), 80.0) * ctx["track"][i],
                    )
                )
            index = np.repeat(np.arange(len(points) - 1), 4)
            blend = np.tile([0, 0.25, 0.5, 0.75], len(points) - 1)
            index, blend = np.r_[index, len(points) - 2], np.r_[blend, 1.0]
            costs.append(
                geometry(
                    robot.joint_var_cls(jnp.asarray(index)),
                    robot.joint_var_cls(jnp.asarray(index + 1)),
                    jnp.asarray(blend[:, None]),
                    ctx["scene"][0][None],
                    ctx["scene"][1][None],
                )
            )
            return costs

        self.solver = make_leg_solver(
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
            max_iterations=320,
        )

    def solve(self, start, goal, scene, moving, track, held, cube, label):
        started = time.monotonic()
        u = np.linspace(0, 1, N_WAYPOINTS)
        initial = start + u[:, None] * (goal - start)
        endpoints = np.asarray(self.fk(jnp.asarray(np.stack((start, goal)))))[:, self.tcp]
        targets = np.empty((2, N_WAYPOINTS, 7), np.float32)
        for i in range(2):
            a, b = (jaxlie.SO3(jnp.asarray(p[:4])) for p in endpoints[:, i])
            targets[i, :, :4] = np.asarray(
                (a @ jaxlie.SO3.exp(jnp.asarray(u[:, None]) * (a.inverse() @ b).log())).wxyz
            )
            targets[i, :, 4:] = endpoints[0, i, 4:] + u[:, None] * (
                endpoints[1, i, 4:] - endpoints[0, i, 4:]
            )
        data = mujoco.MjData(self.model)
        data.qpos[:] = scene
        owners = [(a, c) for a, c in held.items() if c is not None]
        if len(owners) > 1:
            raise ValueError(
                "Motion is forbidden during dual ownership; preserve the handoff barrier"
            )
        carrier, payload = owners[0] if owners else ("left", None)
        states = np.stack(
            [
                g.context(data, allowed_cube=cube, carried_cube=payload, carrier_arm=carrier)
                for g in self.geometry
            ]
        )
        moving = np.repeat(np.asarray(moving, dtype=np.float32), 6)
        context = dict(
            start=jnp.asarray(start),
            moving=jnp.asarray(moving),
            targets=jnp.asarray(targets),
            track=jnp.asarray(track),
            scene=jnp.asarray(states),
        )
        seed = initial
        rejected = []
        for attempt in range(2):
            try:
                result = self.solver(start, goal, initial_waypoints=seed, context=context)
                break
            except InfeasibleTrajectoryError as exc:
                if attempt or exc.positions is None or not np.isfinite(exc.positions).all():
                    raise
                rejected.append(exc.max_constraint_violation)
                seed = np.asarray(exc.positions).copy()
                seed[:, moving == 0] = start[moving == 0]
                print(
                    f"[joint/{self.variant}] {label}: retry from rejected candidate "
                    f"(violation {exc.max_constraint_violation:.4g})",
                    flush=True,
                )
        q = np.asarray(result.positions).copy()
        idle_correction = float(np.abs((q - start) * (1 - moving)).max())
        q[:, moving == 0] = start[moving == 0]
        poses = np.asarray(self.fk(jnp.asarray(q)))[:, self.tcp, 4:]
        deviations = np.linalg.norm(poses.transpose(1, 0, 2) - targets[:, :, 4:], axis=-1).max(
            axis=1
        )
        if np.any(np.asarray(track) & (deviations > 0.015)):
            raise RuntimeError("A Cartesian handoff leg exceeds the 15 mm tracking tolerance")
        self.states = states
        self.validate(q)
        record = dict(
            label=label,
            variant=self.variant,
            style_cost=float(self.encoder.cost(jnp.asarray(q), SOURCE_DT)),
            initial_style_cost=float(self.encoder.cost(jnp.asarray(initial), SOURCE_DT)),
            max_constraint_violation=result.max_constraint_violation,
            idle_projection_rad=idle_correction,
            planning_seconds=time.monotonic() - started,
            moving=moving.tolist(),
            tracking=list(map(bool, track)),
            max_cartesian_reference_deviation_m=deviations.tolist(),
            rejected_constraint_violations=rejected,
        )
        self.records.append(record)
        self.source_paths.append(q.copy())
        print(f"[joint/{self.variant}] {label}: style {record['style_cost']:.3f}", flush=True)
        return q, record

    def validate(self, q):
        # Denser samples also check the exact idle-arm projection.
        grid = np.linspace(0, len(q) - 1, 5 * (len(q) - 1) + 1)
        dense = np.stack([np.interp(grid, np.arange(len(q)), c) for c in q.T], axis=1)
        return min(
            g.validate_path(dense, state)
            for g, state in zip(self.geometry, self.states, strict=True)
        )

    def payload_pose(self, q, cube):
        self.geometry[0].evaluate(q, self.states[0])
        adr = self.model.joint(f"cube{cube}_free").qposadr[0]
        return self.geometry[0].data.qpos[adr : adr + 7].copy()
