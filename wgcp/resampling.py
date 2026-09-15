"""Compositing an evaluation pool to a target spurious-correlation strength.

Groups are g = 2y + a; a sample is aligned when a == y, i.e. g in {0, 3}. The correlation
strength is rho = P(a == y) at balanced classes, so the target group fractions are

    g0 = g3 = rho / 2,    g1 = g2 = (1 - rho) / 2.

Calibration and test pools are disjoint, and each is drawn with replacement within its own
group indices so that an extreme rho stays reachable from a finite pool.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GROUPS = (0, 1, 2, 3)
ALIGNED_GROUPS = (0, 3)
MINORITY_GROUPS = (1, 2)


def target_group_fractions(rho: float) -> np.ndarray:
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0,1], got {rho}")
    return np.array([rho / 2.0, (1.0 - rho) / 2.0, (1.0 - rho) / 2.0, rho / 2.0], dtype=np.float64)


def realized_rho(group_id: np.ndarray) -> float:
    g = np.asarray(group_id)
    if g.size == 0:
        return float("nan")
    return float(np.isin(g, ALIGNED_GROUPS).mean())


def _target_counts(rho: float, n: int) -> np.ndarray:
    """Largest-remainder rounding, so the counts sum to exactly n."""
    frac = target_group_fractions(rho)
    raw = frac * n
    counts = np.floor(raw).astype(int)
    remainder = n - counts.sum()
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for k in range(remainder):
            counts[order[k % len(counts)]] += 1
    return counts


def split_pool(n_total: int, frac_cal: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded disjoint split of pooled indices into (cal_pool, test_pool)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_cal = int(round(frac_cal * n_total))
    return np.sort(perm[:n_cal]), np.sort(perm[n_cal:])


@dataclass
class ResampleResult:
    idx: np.ndarray
    counts: np.ndarray
    target_counts: np.ndarray
    rho_target: float
    rho_realized: float


def resample_to_rho(group_id_pool: np.ndarray, rho: float, n: int, seed: int,
                    replace: bool = True) -> ResampleResult:
    """Indices into ``group_id_pool`` realising rho at total size n."""
    rng = np.random.default_rng(seed)
    g = np.asarray(group_id_pool)
    counts = _target_counts(rho, n)
    chosen = []
    for grp in GROUPS:
        want = int(counts[grp])
        if want == 0:
            continue
        members = np.where(g == grp)[0]
        if members.size == 0:
            raise ValueError(f"pool has no samples in group {grp}; cannot hit rho={rho}")
        if (not replace) and want > members.size:
            raise ValueError(f"group {grp}: need {want} but only {members.size} available "
                             f"(set replace=True for extreme rho)")
        chosen.append(rng.choice(members, size=want, replace=replace))
    idx = np.concatenate(chosen)
    rng.shuffle(idx)
    realized_counts = np.array([(g[idx] == grp).sum() for grp in GROUPS], dtype=int)
    return ResampleResult(idx=idx, counts=realized_counts, target_counts=counts,
                          rho_target=float(rho), rho_realized=realized_rho(g[idx]))
