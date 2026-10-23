"""Model identity and explicit mappings between training and robot joint order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import numpy as np


@dataclass(frozen=True)
class EncoderSpec:
    """Static model contract. Names describe the ordered channels used during training."""

    name: str
    joint_names: tuple[str, ...]
    sample_rate_hz: float = 15.0

    def __post_init__(self):
        if not self.name or not self.joint_names or any(not x for x in self.joint_names):
            raise ValueError("A model needs a name and a nonempty ordered list of joint names")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Model joint names must be unique")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive and finite")

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)


@dataclass(frozen=True)
class JointMap:
    """Indices extracting model-ordered coordinates from a robot configuration.

    Construct from names, not an inferred matching DOF count. The same number of
    joints does not imply that a model was trained for a particular robot.
    """

    robot_joint_names: tuple[str, ...]
    model_joint_names: tuple[str, ...]
    indices: tuple[int, ...]

    def __post_init__(self):
        if len(set(self.model_joint_names)) != len(self.model_joint_names):
            raise ValueError("Model joint names must be unique")
        if len(set(self.robot_joint_names)) != len(self.robot_joint_names):
            raise ValueError("Robot joint names must be unique")
        if len(self.indices) != len(self.model_joint_names) or not self.indices:
            raise ValueError("One index per model joint is required")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("Each model joint must select a distinct robot coordinate")
        if any(i < 0 or i >= len(self.robot_joint_names) for i in self.indices):
            raise ValueError("Joint index out of range")

    @classmethod
    def from_names(
        cls,
        robot_joint_names: Sequence[str],
        model_joint_names: Sequence[str],
        *,
        aliases: Mapping[str, str] | None = None,
    ) -> JointMap:
        """aliases maps model names to URDF names; missing names are errors."""
        robot_names = tuple(robot_joint_names)
        model_names = tuple(model_joint_names)
        aliases = {} if aliases is None else aliases
        unknown = set(aliases) - set(model_names)
        if unknown:
            raise ValueError(f"Aliases contain unknown model joints: {sorted(unknown)}")
        targets = tuple(aliases.get(name, name) for name in model_names)
        missing = set(targets) - set(robot_names)
        if missing:
            raise ValueError(
                f"Robot is missing model joints {sorted(missing)}; supply explicit aliases"
            )
        return cls(robot_names, model_names, tuple(robot_names.index(name) for name in targets))

    def select(self, q: jax.Array) -> jax.Array:
        """Select/reorder the final axis; compatible with JIT and autodiff."""
        if q.shape[-1] != len(self.robot_joint_names):
            raise ValueError("Configuration width does not match the robot's joint names")
        return q[..., np.asarray(self.indices)]
