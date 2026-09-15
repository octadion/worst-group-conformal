"""Per-group and worst-group accuracy.

Group ids follow g = 2y + a for label y and binary attribute a, so g0 and g3 are the cells
where the attribute agrees with the label.
"""
from __future__ import annotations

import numpy as np

__all__ = ["group_ids", "per_group_accuracy", "worst_group_accuracy"]


def group_ids(y: np.ndarray, place: np.ndarray) -> np.ndarray:
    return (2 * np.asarray(y).astype(int) + np.asarray(place).astype(int)).astype(int)


def per_group_accuracy(y_pred: np.ndarray, y_true: np.ndarray, group_id: np.ndarray) -> dict:
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    g = np.asarray(group_id)
    correct = (y_pred == y_true)
    return {int(grp): float(correct[g == grp].mean()) for grp in np.unique(g)}


def worst_group_accuracy(y_pred: np.ndarray, y_true: np.ndarray,
                         group_id: np.ndarray) -> tuple[int, float]:
    """Ties are broken by the smallest group id."""
    acc = per_group_accuracy(y_pred, y_true, group_id)
    worst_g = min(acc, key=lambda g: (acc[g], g))
    return int(worst_g), float(acc[worst_g])
