"""Signed MuJoCo geometry distances and witness-point derivatives for local planning.

This CPU example uses the same collision primitives as replay, including the fingers
and wrist camera. JAX receives analytic distance Jacobians through a host callback.
Attached objects here are planning geometry only; simulation uses free-body contacts.
"""

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from _yam_scene import ARM_NAMES, JOINT_NAMES


class SceneDistances:
    def __init__(self, model, arm, *, coupled=False):
        self.model, self.arm = model, arm
        self.data = mujoco.MjData(model)
        names = JOINT_NAMES if coupled else ARM_NAMES[arm]
        self.qpos = np.array([model.joint(n).qposadr[0] for n in names])
        self.dofs = np.array([model.joint(n).dofadr[0] for n in names])
        self.site = model.site(f"{arm}_grasp_site").id
        self.flange = model.body(f"{arm}_link_6").id
        bodies = [model.body(int(b)).name for b in model.geom_bodyid]
        physical = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
        active = [g for g in physical if bodies[g].startswith(arm + "_")]
        other = [
            g for g in physical if bodies[g].startswith(("left_", "right_")) and g not in active
        ]
        self.cubes = [g for g in physical if bodies[g].startswith("cube")]
        self.obstacles = [model.geom("table").id, model.geom("tray").id, *self.cubes]
        self.active = active
        self.fingers = {g for g in physical if "finger" in bodies[g] or "pad" in bodies[g]}
        self.finger_joints = {
            g: model.joint(f"{bodies[g].split('_', 1)[0]}_left_finger").qposadr[0]
            for g in self.fingers
        }
        excluded = {int(s) for s in model.exclude_signature}

        def eligible(a, b):
            ba, bb = sorted((int(model.geom_bodyid[a]), int(model.geom_bodyid[b])))
            wa, wb = int(model.body_weldid[ba]), int(model.body_weldid[bb])
            if wa == wb or (ba << 16 | bb) in excluded:
                return False
            if (
                wa == model.body_weldid[model.body_parentid[wb]]
                or wb == model.body_weldid[model.body_parentid[wa]]
            ):
                return False
            return bool(
                model.geom_contype[a] & model.geom_conaffinity[b]
                or model.geom_contype[b] & model.geom_conaffinity[a]
            )

        self.groups = [[(a, b) for a in active if eligible(a, b)] for b in self.obstacles]
        self.groups += [
            [(a, b) for a, b in itertools.product(active, other) if eligible(a, b)],
            [(a, b) for a, b in itertools.combinations(active, 2) if eligible(a, b)],
        ]
        # Separate carried-object constraints prevent a cube clipping the table,
        # tray, other cubes, or the other hand while its TCP is clear.
        self.groups += [[(a, b) for a in self.cubes] for b in self.obstacles]
        self.groups += [[(a, b) for a, b in itertools.product(self.cubes, other)]]
        self.groups = [np.asarray(g, dtype=int).reshape(-1, 2) for g in self.groups]
        self.count = len(self.groups)
        self._cache = OrderedDict()
        self._lock = threading.RLock()

        @jax.custom_jvp
        def distance(q, state):
            return callback(q, state)[0]

        def callback(q, state):
            return jax.pure_callback(
                self.evaluate,
                (
                    jax.ShapeDtypeStruct((self.count,), q.dtype),
                    jax.ShapeDtypeStruct((self.count, len(names)), q.dtype),
                ),
                q,
                state,
                vmap_method="sequential",
            )

        @distance.defjvp
        def distance_jvp(primals, tangents):
            q, state = primals
            dq, _ = tangents
            values, jacobian = callback(q, state)
            return values, jacobian @ dq

        self.distance = distance

    def context(self, data, *, allowed_cube=None, carried_cube=None, carrier_arm=None):
        """Snapshot all poses and the rigid TCP-to-cube transform for this leg."""
        mujoco.mj_forward(self.model, data)
        carrier_arm = self.arm if carrier_arm is None else carrier_arm
        site = self.model.site(f"{carrier_arm}_grasp_site").id
        offset, rotation = np.zeros(3), np.eye(3)
        if carried_cube is not None:
            body = self.model.body(f"cube{carried_cube}").id
            tcp_rotation = data.site_xmat[site].reshape(3, 3)
            offset = tcp_rotation.T @ (data.xpos[body] - data.site_xpos[site])
            rotation = tcp_rotation.T @ data.xmat[body].reshape(3, 3)
        return np.r_[
            data.qpos,
            -1 if allowed_cube is None else allowed_cube,
            -1 if carried_cube is None else carried_cube,
            offset,
            rotation.ravel(),
            int(carrier_arm == "right"),
        ].astype(np.float32)

    def evaluate(self, q, state):
        with self._lock:
            return self._evaluate(q, state)

    def validate_path(self, positions, state):
        """Check the interpolated/retimed commands, including payload geometry."""
        minimum = min(float(self.evaluate(row, state)[0].min()) for row in positions)
        if minimum < -0.0005:
            raise RuntimeError(
                f"Interpolated {self.arm} path penetrates geometry by {-minimum:g} m"
            )
        return minimum

    def _evaluate(self, q, state):
        q, state = np.asarray(q), np.asarray(state)
        key = (q.tobytes(), state.tobytes())
        if key in self._cache:
            result, qpos = self._cache[key]
            self._cache.move_to_end(key)
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            return result
        model, data = self.model, self.data
        data.qpos[:] = state[: model.nq]
        allowed, carried = state[model.nq : model.nq + 2].astype(int)
        data.qpos[self.qpos] = q
        mujoco.mj_forward(model, data)
        carried_geom = self.cubes[carried] if carried >= 0 else -1
        carrier_flange = self.flange
        if carried >= 0:
            tail = state[model.nq + 2 :]
            carrier_arm = "right" if int(tail[12]) else "left"
            carrier_site = model.site(f"{carrier_arm}_grasp_site").id
            carrier_flange = model.body(f"{carrier_arm}_link_6").id
            rot = data.site_xmat[carrier_site].reshape(3, 3)
            adr = model.joint(f"cube{carried}_free").qposadr[0]
            data.qpos[adr : adr + 3] = data.site_xpos[carrier_site] + rot @ tail[:3]
            mujoco.mju_mat2Quat(
                data.qpos[adr + 3 : adr + 7], (rot @ tail[3:12].reshape(3, 3)).ravel()
            )
            mujoco.mj_forward(model, data)
        values = np.full(self.count, 0.06, q.dtype)
        jacobian = np.zeros((self.count, len(self.dofs)), q.dtype)
        fromto = np.empty(6)
        jac = np.empty((3, model.nv))
        for group, pairs in enumerate(self.groups):
            nearest = None
            if not len(pairs):
                continue
            lower_bound = (
                np.linalg.norm(data.geom_xpos[pairs[:, 0]] - data.geom_xpos[pairs[:, 1]], axis=1)
                - model.geom_rbound[pairs[:, 0]]
                - model.geom_rbound[pairs[:, 1]]
            )
            pairs = pairs[lower_bound < values[group]]
            for a, b in pairs:
                if group >= len(self.obstacles) + 2 and (a != carried_geom or a == b):
                    continue
                if allowed >= 0 and (
                    (
                        a == self.cubes[allowed]
                        and b in self.fingers
                        and data.qpos[self.finger_joints[b]] < 0.03
                    )
                    or (
                        b == self.cubes[allowed]
                        and a in self.fingers
                        and data.qpos[self.finger_joints[a]] < 0.03
                    )
                ):
                    continue
                if (
                    np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b])
                    - model.geom_rbound[a]
                    - model.geom_rbound[b]
                    >= values[group]
                ):
                    continue
                dist = mujoco.mj_geomDistance(model, data, a, b, float(values[group]), fromto)
                if dist < values[group]:
                    values[group] = dist
                    nearest = (a, b, fromto.copy())
            if nearest is None:
                continue
            a, b, segment = nearest
            normal = segment[3:] - segment[:3]
            length = np.linalg.norm(normal)
            if length < 1e-9:
                continue
            normal *= (1 if values[group] >= 0 else -1) / length
            for geom, point, sign in ((a, segment[:3], -1), (b, segment[3:], 1)):
                body = carrier_flange if geom == carried_geom else int(model.geom_bodyid[geom])
                mujoco.mj_jac(model, data, jac, None, point, body)
                jacobian[group] += sign * normal @ jac[:, self.dofs]
        result = values, jacobian
        self._cache[key] = result, data.qpos.copy()
        if len(self._cache) > 4096:
            self._cache.popitem(last=False)
        return result

    @property
    def margins(self):
        margin = np.full(self.count, 0.003, np.float32)
        # Cover tracking error and small grasp/placement offsets around movable blocks.
        margin[2 : len(self.obstacles)] = 0.020
        payload = len(self.obstacles) + 2
        margin[payload + 2 : payload + len(self.obstacles)] = 0.020
        # The model's neighboring wrist parts have only a few mm of design clearance.
        margin[len(self.obstacles) + 1] = 0.001
        # Resting carried cubes may touch their supporting table/tray at endpoints.
        margin[len(self.obstacles) + 2 : len(self.obstacles) + 4] = 0.0
        return jnp.asarray(margin)
