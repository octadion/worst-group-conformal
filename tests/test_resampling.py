"""Compositing hits the target correlation strength, and the two halves stay disjoint."""
import numpy as np

from wgcp.resampling import realized_rho, resample_to_rho, split_pool


def _pool(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    a = np.where(rng.random(n) < 0.6, y, 1 - y)
    return 2 * y + a


def test_composited_pool_hits_the_target_rho():
    g = _pool()
    for rho in (0.95, 0.5, 0.25):
        rs = resample_to_rho(g, rho, 2000, seed=1)
        assert abs(rs.rho_realized - rho) < 0.02, f"rho {rho}: realised {rs.rho_realized:.3f}"
        assert rs.idx.size == 2000


def test_group_fractions_follow_rho():
    g = _pool()
    rs = resample_to_rho(g, 0.95, 4000, seed=2)
    frac = rs.counts / rs.counts.sum()
    assert np.allclose(frac, [0.475, 0.025, 0.025, 0.475], atol=0.01)


def test_calibration_and_test_pools_are_disjoint():
    cal, test = split_pool(5000, 0.5, seed=3)
    assert set(cal.tolist()).isdisjoint(test.tolist())
    assert cal.size + test.size == 5000


def test_realized_rho_counts_the_aligned_groups():
    assert realized_rho(np.array([0, 3, 0, 3])) == 1.0
    assert realized_rho(np.array([1, 2, 1, 2])) == 0.0
