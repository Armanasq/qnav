"""Robotics (ROS REP-103/REP-105 style) frame helpers.

Standard frames: ``map``/``odom`` (ENU-aligned world), ``base_link`` (FLU
body), plus sensor frames. Quaternion layout bridges to ROS message order
(scalar-last) are provided here so call sites never reorder by hand.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.frames.core import Frame
from qnav.frames.graph import FrameGraph
from qnav.frames.transforms import FrameTransform

__all__ = [
    "MAP", "ODOM", "BASE_LINK", "standard_graph",
    "quaternion_to_ros", "quaternion_from_ros",
]

MAP = Frame("map", "x:east y:north z:up (ENU-aligned)", "local_tangent")
ODOM = Frame("odom", "x:east y:north z:up (ENU-aligned, drifting)", "local_tangent")
BASE_LINK = Frame("base_link", "x:forward y:left z:up", "body")


def standard_graph() -> FrameGraph:
    """A graph pre-wired with identity ``map → odom → base_link`` placeholders.

    Replace edges with live transforms via ``graph.add(..., replace=True)``.
    """
    g = FrameGraph()
    g.add(FrameTransform(target="map", source="odom", rotation=quat.identity()))
    g.add(FrameTransform(target="odom", source="base_link", rotation=quat.identity()))
    return g


def quaternion_to_ros(q: np.ndarray) -> np.ndarray:
    """qnav ``[w, x, y, z]`` → ROS message order ``[x, y, z, w]`` (same rotation,
    both Hamilton)."""
    return quat.to_scalar_last(q)


def quaternion_from_ros(q_ros: np.ndarray) -> np.ndarray:
    """ROS ``[x, y, z, w]`` → qnav ``[w, x, y, z]``."""
    return quat.from_scalar_last(q_ros)
