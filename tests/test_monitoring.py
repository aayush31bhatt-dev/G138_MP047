"""Unit tests for drift PSI calculation and model promotion logic."""

from __future__ import annotations

import numpy as np

from src.bike_price.monitoring import compute_psi, should_promote


def test_compute_psi_identical_and_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(10.0, 2.0, size=500)
    same = rng.normal(10.0, 2.0, size=500)
    shifted = rng.normal(18.0, 2.0, size=500)

    psi_same = compute_psi(ref, same, bins=10)
    psi_shifted = compute_psi(ref, shifted, bins=10)

    assert psi_same < 0.1
    assert psi_shifted > 1.0


def test_should_promote_threshold():
    # Exact 8% improvement -> promote
    assert should_promote(10.0, 9.2, min_improvement=0.08) is True
    # Only 5% improvement -> keep
    assert should_promote(10.0, 9.5, min_improvement=0.08) is False
