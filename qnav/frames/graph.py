"""Frame-transform graph: register transforms, look up composed paths.

The graph stores :class:`~qnav.frames.transforms.FrameTransform` edges between
named frames. Queries find the (unique shortest) path between two frames and
return the fully composed transform — direction handled automatically via
inverses. Cycles are allowed only if consistent; adding a transform between
already-connected frames raises unless ``replace=True``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from qnav.errors import FrameGraphError
from qnav.frames.transforms import FrameTransform

__all__ = ["FrameGraph"]


class FrameGraph:
    """A registry of frame transforms supporting path lookup and composition."""

    def __init__(self) -> None:
        # adjacency: frame -> {neighbor: FrameTransform(target=neighbor? ...)}
        # we store each edge once as given; orientation resolved at query time
        self._edges: Dict[str, Dict[str, FrameTransform]] = {}

    @property
    def frames(self) -> Tuple[str, ...]:
        """All frame names currently in the graph."""
        return tuple(sorted(self._edges))

    def add(self, transform: FrameTransform, *, replace: bool = False) -> None:
        """Register a transform edge between ``transform.source`` and ``.target``.

        Raises :class:`FrameGraphError` if an edge between the two frames
        already exists (in either direction) and ``replace`` is False, or if
        the new edge would create a second path between already-connected
        frames (ambiguous composition).
        """
        a, b = transform.target, transform.source
        if a == b:
            raise FrameGraphError("self-transforms are not allowed in the graph")
        existing = self._edges.get(a, {}).get(b)
        if existing is not None and not replace:
            raise FrameGraphError(
                f"an edge between {a!r} and {b!r} already exists; use replace=True"
            )
        if existing is None and a in self._edges and b in self._edges:
            if self._find_path(a, b) is not None and not replace:
                raise FrameGraphError(
                    f"frames {a!r} and {b!r} are already connected via another "
                    f"path; adding this edge would make composition ambiguous"
                )
        self._edges.setdefault(a, {})[b] = transform
        self._edges.setdefault(b, {})[a] = transform  # same edge, resolved on query

    def _find_path(self, start: str, goal: str) -> Optional[List[str]]:
        """BFS shortest path by frame names; None if unreachable."""
        if start not in self._edges or goal not in self._edges:
            return None
        prev: Dict[str, Optional[str]] = {start: None}
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node == goal:
                path = [node]
                while prev[node] is not None:
                    node = prev[node]  # type: ignore[assignment]
                    path.append(node)
                return path[::-1]
            for nb in self._edges[node]:
                if nb not in prev:
                    prev[nb] = node
                    queue.append(nb)
        return None

    def get(self, target: str, source: str) -> FrameTransform:
        """Composed transform ``T_target_source`` along the shortest path.

        Edges traversed against their stored direction are inverted. Raises
        :class:`FrameGraphError` if either frame is unknown or no path exists.
        """
        if target == source:
            if target not in self._edges:
                raise FrameGraphError(f"unknown frame {target!r}")
            return FrameTransform.identity(target)
        path = self._find_path(source, goal=target)
        if path is None:
            known = ", ".join(self.frames) or "<empty graph>"
            raise FrameGraphError(
                f"no path from {source!r} to {target!r}; known frames: {known}"
            )
        # walk source -> target accumulating T_target_source as successive lefts
        result: Optional[FrameTransform] = None
        for u, v in zip(path[:-1], path[1:]):
            edge = self._edges[u][v]
            step = edge if (edge.source == u and edge.target == v) else edge.inverse()
            result = step if result is None else step @ result
        assert result is not None
        return result

    def transform_vector(self, v, target: str, source: str):
        """Rotate free vector(s) from ``source`` to ``target`` coordinates."""
        return self.get(target, source).apply_vector(v)

    def transform_point(self, p, target: str, source: str):
        """Transform point(s) from ``source`` to ``target`` coordinates."""
        return self.get(target, source).apply_point(p)
