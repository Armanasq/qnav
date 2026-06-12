"""Frame identity and registry.

A :class:`Frame` is a lightweight named token carrying its axis convention as
documentation-grade metadata. Transforms (``qnav.frames.transforms``) are
checked against frame names at runtime, eliminating the classic
"applied body-to-nav where nav-to-body was needed" bug class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qnav.errors import ConventionError

__all__ = ["Frame", "WELL_KNOWN_FRAMES"]


@dataclass(frozen=True)
class Frame:
    """A named right-handed coordinate frame.

    Attributes
    ----------
    name:
        Unique identifier used for transform checking (e.g. ``"NED"``,
        ``"body_frd"``, ``"imu0"``).
    axes:
        Human-readable axis convention, e.g. ``"x:north y:east z:down"``.
    kind:
        One of ``{"earth", "local_tangent", "body", "sensor", "other"}``.
    """

    name: str
    axes: str = ""
    kind: str = "other"

    _KINDS = ("earth", "local_tangent", "body", "sensor", "other")

    def __post_init__(self) -> None:
        if not self.name:
            raise ConventionError("frame name must be non-empty")
        if self.kind not in self._KINDS:
            raise ConventionError(f"frame kind must be one of {self._KINDS}, got {self.kind!r}")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


def _wk(name: str, axes: str, kind: str) -> Frame:
    return Frame(name=name, axes=axes, kind=kind)


#: Frames qnav knows by name. Users may define any additional frames.
WELL_KNOWN_FRAMES: dict[str, Frame] = {
    "ECI": _wk("ECI", "x:vernal-equinox y:RH-complement z:earth-spin-axis", "earth"),
    "ECEF": _wk("ECEF", "x:0N0E y:0N90E z:north-pole", "earth"),
    "NED": _wk("NED", "x:north y:east z:down", "local_tangent"),
    "ENU": _wk("ENU", "x:east y:north z:up", "local_tangent"),
    "FRD": _wk("FRD", "x:forward y:right z:down", "body"),
    "FLU": _wk("FLU", "x:forward y:left z:up", "body"),
}
