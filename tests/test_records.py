"""The shipped records still produce the numbers the paper reports.

These are regression tests over ``records/``: if the analysis code drifts, the anchors move.
"""
import numpy as np
import pytest

from wgcp.intervals import table2

ANCHORS = {                       # (shared, per-group, lift), as printed in the paper's Table 2
    ("waterbirds", "resnet50_erm"): (0.564, 0.870, 0.306),
    ("waterbirds", "clip_vitb32"): (0.509, 0.854, 0.345),
    ("waterbirds", "dinov2_vitb14"): (0.818, 0.881, 0.063),
    ("waterbirds", "vit_b16_in1k"): (0.647, 0.865, 0.218),
    ("celeba", "resnet50_erm"): (0.821, 0.880, 0.059),
    ("celeba", "clip_vitb32"): (0.822, 0.886, 0.063),
    ("celeba", "dinov2_vitb14"): (0.813, 0.885, 0.072),
    ("celeba", "vit_b16_in1k"): (0.804, 0.886, 0.082),
}


@pytest.fixture(scope="module")
def rows():
    return {(r["dataset"], r["backbone"]): r for r in table2(B=200)}


def test_every_setting_matches_the_reported_means(rows):
    for key, (shared, per_group, lift) in ANCHORS.items():
        r = rows[key]
        assert round(r["shared"], 3) == shared, key
        assert round(r["per_group"], 3) == per_group, key
        assert round(r["lift"], 3) == lift, key


def test_the_shared_threshold_spreads_far_more_than_the_per_group_one(rows):
    shared = [r["spread_shared"] for r in rows.values()]
    per_group = [r["spread_per_group"] for r in rows.values()]
    assert max(shared) > 0.35, "the shared-threshold spread should reach 0.358"
    assert max(per_group) <= 0.025, "the per-group spread should stay at or below 0.024"
    larger = sum(s > p for s, p in zip(shared, per_group))
    assert larger == 7, f"the shared spread should be the larger one in seven settings, got {larger}"


def test_the_lift_intervals_exclude_zero(rows):
    for key, r in rows.items():
        lo, hi = r["lift_ci"]
        assert lo > 0, f"{key}: lift interval [{lo:.3f},{hi:.3f}] touches zero"


def test_records_carry_every_cell():
    assert len(ANCHORS) == 8
    assert not np.isnan([r["shared"] for r in table2(B=50)]).any()
