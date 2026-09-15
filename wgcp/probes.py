"""The standardised linear probe every last-layer head is built from.

StandardScaler followed by multinomial logistic regression (lbfgs). Features must be
L2-normalised: an unnormalised linear probe on CLIP features under-fits badly.
"""
from __future__ import annotations

import numpy as np

__all__ = ["assert_l2_normalized", "fit_species_head", "head_probs", "probe_posteriors", "top1"]


def assert_l2_normalized(X: np.ndarray, tag: str = "features", tol: float = 1e-2) -> None:
    """Row norms are computed in chunks: the one-shot expression holds a full float64 copy."""
    X = np.asarray(X)
    norms = np.empty(X.shape[0], dtype=np.float64)
    for i in range(0, X.shape[0], 2048):
        blk = X[i:i + 2048]
        norms[i:i + 2048] = np.linalg.norm(np.asarray(blk, dtype=np.float64), axis=1)
    if X.shape[0] and not np.allclose(norms, 1.0, atol=tol):
        raise ValueError(f"{tag} are not L2-normalised (mean norm {norms.mean():.3f}, "
                         f"min {norms.min():.3f}, max {norms.max():.3f})")


def fit_species_head(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0,
                     max_iter: int = 5000, seed: int = 0):
    """Returned as a Pipeline, so ``predict_proba`` and ``classes_`` work directly."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=max_iter, random_state=seed))
    clf.fit(X_train, y_train)
    return clf


def probe_posteriors(clf, X: np.ndarray, n_classes: int) -> np.ndarray:
    """(N, n_classes) posteriors; a class absent from training gets a zero column."""
    p = clf.predict_proba(X)
    classes = np.asarray(clf.classes_).astype(int)
    if classes.shape[0] == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return p
    full = np.zeros((X.shape[0], n_classes), dtype=np.float64)
    full[:, classes] = p
    return full


def head_probs(clf, X: np.ndarray, n_classes: int) -> np.ndarray:
    return probe_posteriors(clf, X, n_classes)


def top1(clf, X: np.ndarray, y: np.ndarray) -> float:
    return float((clf.predict(X) == y).mean())
