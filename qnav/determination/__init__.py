"""Vector-observation attitude determination (Wahba problem solvers).

Common conventions: observations are unit-vector pairs ``(v_ref, v_body)``
with ``v_ref ≈ R_ref_body · v_body``; solvers return ``q_ref_body``
(scalar-first Hamilton) or ``R_ref_body``.

Solver selection guide:

- :func:`~qnav.determination.triad.triad` — 2 vectors, deterministic, fastest.
- :func:`~qnav.determination.davenport.davenport` — optimal, robust (eigen).
- :func:`~qnav.determination.quest.quest` — optimal, fast (Newton), exact
  Davenport fallback near degeneracy.
- :func:`~qnav.determination.svd.svd_attitude` — optimal, best diagnostics.
- :func:`~qnav.determination.oleq.oleq` — linear least-squares formulation.

ESOQ/ESOQ2 are deliberately not implemented: the indexed sources in
``__data`` do not cover them in implementable detail, and qnav's
traceability policy (``docs/conventions.md`` §11) forbids from-memory
implementations. QUEST + Davenport cover the same use cases.
"""

from qnav.determination import davenport, oleq, quest, svd, triad, wahba  # noqa: F401
from qnav.determination.davenport import davenport as davenport_q  # noqa: F401
from qnav.determination.oleq import oleq as oleq_q  # noqa: F401
from qnav.determination.quest import quest as quest_q  # noqa: F401
from qnav.determination.svd import svd_attitude  # noqa: F401
from qnav.determination.triad import triad as triad_dcm  # noqa: F401

__all__ = [
    "davenport", "oleq", "quest", "svd", "triad", "wahba",
    "davenport_q", "oleq_q", "quest_q", "svd_attitude", "triad_dcm",
]
