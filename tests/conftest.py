"""Shared test fixtures and tolerance policy."""

import numpy as np
import pytest

# Tolerance policy (documented in docs/design/api_principles.md):
# - algebraic identities at float64: 1e-12
# - iterative/trig round trips: 1e-9
# - finite-difference Jacobian checks: 1e-5 (eps = 1e-7)
TOL_ALG = 1e-12
TOL_NUM = 1e-9
TOL_FD = 1e-5


@pytest.fixture
def rng():
    return np.random.default_rng(20260612)
