"""Vector-observation attitude determination (Wahba problem solvers).

Common conventions: observations are unit-vector pairs ``(v_ref, v_body)``
with ``v_ref ≈ R_ref_body · v_body``; solvers return ``q_ref_body``
(scalar-first Hamilton) or ``R_ref_body``. The accelerometer/magnetometer
specializations take the **specific force** ``f_body`` and field ``m_body``
directly and return ``q_NB``.

Solver selection guide
----------------------
General N-vector Wahba solvers:

- :func:`~qnav.determination.triad.triad` — 2 vectors, deterministic, fastest.
- :func:`~qnav.determination.davenport.davenport` — optimal, robust (eigen).
- :func:`~qnav.determination.quest.quest` — optimal, fast (Newton), exact
  Davenport fallback near degeneracy.
- :func:`~qnav.determination.svd.svd_attitude` — optimal, best diagnostics.
- :func:`~qnav.determination.oleq.oleq` — linear least-squares formulation.
- :func:`~qnav.determination.flae.flae` — quartic characteristic polynomial;
  fastest optimal N-vector solver (companion roots + Newton polish).

Accelerometer + magnetometer closed forms (NED, no eigendecomposition):

- :func:`~qnav.determination.saam.saam` — one square root; the speed champion.
- :func:`~qnav.determination.famc.famc` — analytic Davenport elimination;
  exposes degeneracy through its pivots.
- :func:`~qnav.determination.fqa.fqa` — factored pitch/roll/yaw quaternions;
  magnetic disturbances provably cannot affect tilt.
"""

from qnav.determination import davenport, famc, flae, fqa, oleq, quest, saam, svd, triad, wahba  # noqa: F401
from qnav.determination.davenport import davenport as davenport_q  # noqa: F401
from qnav.determination.famc import famc as famc_q  # noqa: F401
from qnav.determination.flae import flae as flae_q  # noqa: F401
from qnav.determination.fqa import fqa as fqa_q  # noqa: F401
from qnav.determination.oleq import oleq as oleq_q  # noqa: F401
from qnav.determination.quest import quest as quest_q  # noqa: F401
from qnav.determination.saam import saam as saam_q  # noqa: F401
from qnav.determination.svd import svd_attitude  # noqa: F401
from qnav.determination.triad import triad as triad_dcm  # noqa: F401

__all__ = [
    "davenport", "famc", "flae", "fqa", "oleq", "quest", "saam", "svd",
    "triad", "wahba",
    "davenport_q", "famc_q", "flae_q", "fqa_q", "oleq_q", "quest_q",
    "saam_q", "svd_attitude", "triad_dcm",
]
