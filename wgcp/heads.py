"""ERM and DFR heads on frozen features.

ERM fits the standardised probe on whichever split it is given. DFR averages the posteriors of
``n_subsets`` probes, each fitted on a group-balanced subsample; ``fit_method`` passes it the
held-out reweighting split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sklearn

from .probes import assert_l2_normalized, fit_species_head, head_probs, top1

__all__ = ["assert_multinomial_safe", "fit_erm", "fit_dfr", "DFRHead", "head_probs", "top1",
           "assert_l2_normalized"]

_SKL_VER = tuple(int(x) for x in sklearn.__version__.split(".")[:2])


def assert_multinomial_safe() -> None:
    """lbfgs resolves to the multinomial loss; fail rather than let a version flip it silently."""
    if _SKL_VER < (1, 0):
        raise RuntimeError(f"sklearn {sklearn.__version__} predates the lbfgs multinomial "
                           f"default; pin multi_class='multinomial' before trusting the head")
    if _SKL_VER >= (2, 0):
        raise RuntimeError(f"sklearn {sklearn.__version__} is newer than the 1.5.x this was run "
                           f"against; re-confirm the lbfgs default before running")


def fit_erm(X_train: np.ndarray, y_train: np.ndarray, *, C: float = 1.0, max_iter: int = 5000,
            seed: int = 0):
    assert_multinomial_safe()
    assert_l2_normalized(X_train, tag="ERM train features")
    return fit_species_head(X_train, y_train, C=C, max_iter=max_iter, seed=seed)


@dataclass
class DFRHead:
    members: list
    classes_: np.ndarray
    n_subsets: int
    subset_size_per_group: int

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        acc = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        cls_index = {c: i for i, c in enumerate(self.classes_)}
        for m in self.members:
            p = m.predict_proba(X)
            for j, c in enumerate(m.classes_):
                acc[:, cls_index[c]] += p[:, j]
        return acc / len(self.members)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def fit_dfr(X_train: np.ndarray, y_train: np.ndarray, group_train: np.ndarray, *,
            n_subsets: int = 10, C: float = 1.0, max_iter: int = 5000, seed: int = 0) -> DFRHead:
    """Each subsample draws every group down to the smallest group's count, without replacement.

    The original averages last-layer weights; averaging posteriors is equivalent in expectation
    and survives the per-subset StandardScaler.
    """
    assert_multinomial_safe()
    assert_l2_normalized(X_train, tag="DFR train features")
    y = np.asarray(y_train)
    g = np.asarray(group_train)
    groups = np.unique(g)
    per_group = int(min((g == grp).sum() for grp in groups))
    if per_group == 0:
        raise ValueError("at least one group is empty; cannot balance")

    members = []
    classes_union: set = set()
    for s in range(n_subsets):
        rng = np.random.default_rng(seed * 10_000 + s)
        idx = np.concatenate([
            rng.choice(np.where(g == grp)[0], size=per_group, replace=False) for grp in groups
        ])
        rng.shuffle(idx)
        clf = fit_species_head(X_train[idx], y[idx], C=C, max_iter=max_iter, seed=seed * 10_000 + s)
        members.append(clf)
        classes_union.update(int(c) for c in clf.classes_)

    return DFRHead(members=members, classes_=np.array(sorted(classes_union)), n_subsets=n_subsets,
                   subset_size_per_group=per_group)
