"""Last-layer group-robustness methods on frozen features.

Every method returns a head with ``predict_proba`` and ``classes_``, so the conformal layer
treats them identically.

    erm                 standardised logistic on the training split
    dfr                 group-balanced retrain averaged over subsets, on the reweighting split
    afr                 reweight by (1 - p_ERM(y))^gamma, no group labels, on the same split
    groupdro_ll         softmax head minimising the worst-group loss
    balanced_subsample  one group-balanced draw, then the standardised head
"""
from __future__ import annotations

import numpy as np

from .heads import assert_l2_normalized, assert_multinomial_safe, fit_dfr, fit_erm, fit_species_head

__all__ = ["METHODS", "fit_method", "fit_afr", "fit_groupdro_ll", "fit_balanced_subsample",
           "SoftmaxGroupDRO"]


def fit_afr(X_rw: np.ndarray, y_rw: np.ndarray, *, gamma: float = 2.0, C: float = 1.0,
            max_iter: int = 5000, seed: int = 0):
    """w_i proportional to (1 - p_ERM(y_i | x_i))^gamma, normalised to mean 1."""
    assert_multinomial_safe()
    assert_l2_normalized(X_rw, tag="AFR reweight features")
    erm = fit_species_head(X_rw, y_rw, C=C, max_iter=max_iter, seed=seed)
    p = erm.predict_proba(X_rw)
    cls = list(erm.classes_)
    p_true = np.array([p[i, cls.index(y_rw[i])] for i in range(len(y_rw))])
    w = np.power(np.clip(1.0 - p_true, 1e-6, 1.0), gamma)
    w = w / w.mean()

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=max_iter, random_state=seed))
    clf.fit(X_rw, y_rw, logisticregression__sample_weight=w)
    return clf


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxGroupDRO:
    """Group weights q on the simplex, updated as q_g <- q_g * exp(eta_q * L_g) each step, with a
    gradient step on sum_g q_g * L_g."""

    def __init__(self, mean, std, W, b, classes_):
        self.mean, self.std, self.W, self.b = mean, std, W, b
        self.classes_ = classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = np.asarray(X, dtype=np.float64, copy=True)
        Z -= self.mean
        Z /= self.std
        return _softmax(Z @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def fit_groupdro_ll(X: np.ndarray, y: np.ndarray, group: np.ndarray, *, lr: float = 0.05,
                    eta_q: float = 0.05, l2: float = 1e-3, steps: int = 2000,
                    seed: int = 0) -> SoftmaxGroupDRO:
    """Full-batch gradient descent.

    Standardisation is done in place on one float64 copy, and the column standard deviations are
    streamed with einsum after centring; both keep the arithmetic bit-identical while holding one
    copy of X rather than four.
    """
    assert_l2_normalized(X, tag="GroupDRO features")
    Z = np.asarray(X, dtype=np.float64, copy=True)
    y = np.asarray(y)
    g = np.asarray(group)
    classes_ = np.array(sorted(np.unique(y)))
    cidx = {c: i for i, c in enumerate(classes_)}
    yk = np.array([cidx[v] for v in y])
    n, d = Z.shape
    C = len(classes_)

    mean = Z.mean(axis=0)
    Z -= mean
    std = np.sqrt(np.einsum("ij,ij->j", Z, Z) / n)
    std[std == 0] = 1.0
    Z /= std

    groups = np.array(sorted(np.unique(g)))
    G = len(groups)
    gk = np.array([int(np.where(groups == v)[0][0]) for v in g])
    group_n = np.array([(gk == j).sum() for j in range(G)], dtype=np.float64)

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, C)) * 0.01
    b = np.zeros(C)
    q = np.ones(G) / G

    Y = np.zeros((n, C))
    Y[np.arange(n), yk] = 1.0

    for _ in range(steps):
        P = _softmax(Z @ W + b)
        ce = -np.log(np.clip(P[np.arange(n), yk], 1e-12, 1.0))
        L = np.array([ce[gk == j].mean() if group_n[j] > 0 else 0.0 for j in range(G)])
        q = q * np.exp(eta_q * L)
        q = q / q.sum()
        sw = q[gk] / group_n[gk]
        grad_logits = (P - Y) * sw[:, None]
        gW = Z.T @ grad_logits + l2 * W
        gb = grad_logits.sum(axis=0)
        W -= lr * gW
        b -= lr * gb

    return SoftmaxGroupDRO(mean=mean, std=std, W=W, b=b, classes_=classes_)


def fit_balanced_subsample(X: np.ndarray, y: np.ndarray, group: np.ndarray, *, C: float = 1.0,
                           max_iter: int = 5000, seed: int = 0):
    assert_multinomial_safe()
    assert_l2_normalized(X, tag="balanced-subsample features")
    g = np.asarray(group)
    groups = np.unique(g)
    per = int(min((g == grp).sum() for grp in groups))
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(g == grp)[0], size=per, replace=False)
                          for grp in groups])
    rng.shuffle(idx)
    return fit_species_head(np.asarray(X)[idx], np.asarray(y)[idx], C=C, max_iter=max_iter,
                            seed=seed)


def fit_method(name: str, train: tuple, reweight: tuple, *, seed: int = 0, **hp):
    """``train`` and ``reweight`` are (X, y, group) tuples; each method takes the split it needs."""
    Xtr, ytr, gtr = train
    Xrw, yrw, grw = reweight
    if name == "erm":
        return fit_erm(Xtr, ytr, seed=seed, **hp)
    if name == "dfr":
        return fit_dfr(Xrw, yrw, grw, seed=seed, **hp)
    if name == "afr":
        return fit_afr(Xrw, yrw, seed=seed, **hp)
    if name == "groupdro_ll":
        return fit_groupdro_ll(Xtr, ytr, gtr, seed=seed, **hp)
    if name == "balanced_subsample":
        return fit_balanced_subsample(Xtr, ytr, gtr, seed=seed, **hp)
    raise ValueError(f"unknown method {name!r}; choose from {list(METHODS)}")


METHODS = ("erm", "dfr", "afr", "groupdro_ll", "balanced_subsample")
