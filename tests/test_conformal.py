"""Coverage guarantees on synthetic scores, where the right answer is known in advance."""
import numpy as np

from conformal.group_robust import mondrian_build_sets, mondrian_quantiles
from conformal.scores import draw_randomization, scores_all, true_label_scores
from conformal.split_conformal import build_sets, conformal_quantile, covered


def _synthetic(n=20000, rho=0.95, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    a = np.where(rng.random(n) < rho, y, 1 - y)
    group = 2 * y + a
    # the minority groups are harder, so the per-group thresholds genuinely differ
    logit = rng.normal(2.0, 1.0, n) - 2.0 * (y != a)
    p_true = 1 / (1 + np.exp(-logit))
    probs = np.stack([1 - p_true, p_true], axis=1)
    probs[y == 0] = probs[y == 0][:, ::-1]
    return probs, y, group


def test_marginal_split_covers_at_target():
    probs, y, group = _synthetic()
    half = len(y) // 2
    for score in ("THR", "APS", "RAPS"):
        u = draw_randomization(len(y), seed=1)
        s = scores_all(score, probs, u=u)
        q = conformal_quantile(true_label_scores(s[:half], y[:half]), 0.1)
        cov = covered(build_sets(s[half:], q), y[half:]).mean()
        assert cov >= 0.88, f"{score}: marginal coverage {cov:.3f} far below target"


def test_mondrian_covers_every_group():
    probs, y, group = _synthetic()
    half = len(y) // 2
    u = draw_randomization(len(y), seed=2)
    s = scores_all("APS", probs, u=u)
    gq = mondrian_quantiles(true_label_scores(s[:half], y[:half]), group[:half], 0.1)
    member = mondrian_build_sets(s[half:], group[half:], gq)
    hit = member[np.arange(half), y[half:]]
    for g in np.unique(group[half:]):
        cov = hit[group[half:] == g].mean()
        assert cov >= 0.85, f"group {g}: per-group coverage {cov:.3f}"


def test_mondrian_sets_do_not_depend_on_the_test_label():
    """A set the test label helps build is not a prediction set."""
    probs, y, group = _synthetic()
    half = len(y) // 2
    u = draw_randomization(len(y), seed=3)
    s = scores_all("APS", probs, u=u)
    gq = mondrian_quantiles(true_label_scores(s[:half], y[:half]), group[:half], 0.1)

    g_test = group[half:]
    relabelled = 2 * (1 - y[half:]) + (g_test % 2)          # same attribute, opposite label
    assert np.array_equal(mondrian_build_sets(s[half:], g_test, gq),
                          mondrian_build_sets(s[half:], relabelled, gq))


def test_quantile_falls_back_to_the_full_set_when_n_is_too_small():
    assert conformal_quantile(np.array([0.1, 0.2]), 0.1) == float("inf")
    assert conformal_quantile(np.array([]), 0.1) == float("inf")
