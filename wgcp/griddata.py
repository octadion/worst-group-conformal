"""Assembling one (dataset, backbone) cell: features, labels and groups per split.

    train        fits ERM, GroupDRO-LL and balanced subsampling
    reweight     the held-out split DFR and AFR are fitted on
    eval_domain  d_cal and d_test pooled; the conformal halves are drawn from here

Every split carries the same composited distribution, so the comparison is in-domain, and the
group id is 2y + a throughout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data.base import SplitSpec
from .features import (FROZEN_BACKBONES, clip_features, frozen_features, l2_normalize,
                       resnet50_erm_features)

__all__ = ["GridData", "build_griddata", "release_memory", "rss_gib"]


@dataclass
class GridData:
    backbone: str
    dataset: str
    train: tuple        # (X, y, group)
    reweight: tuple     # (X, y, group)
    eval_domain: tuple  # (X, y, group)
    n_classes: int


def release_memory() -> None:
    """Return freed arenas to the OS between arms.

    Resident memory grew by twice the float32 training matrix per arm on glibc and never fell,
    which is the float64 copy sklearn makes: the generational collector counts objects rather
    than bytes, and the allocator kept the freed block. ``malloc_trim`` is glibc-only, so it is
    attempted and ignored elsewhere. Nothing computed depends on when memory is released.
    """
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def rss_gib() -> float:
    """Resident set size in GiB, or nan; telemetry only, never control flow."""
    import os
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        try:
            with open("/proc/self/statm") as fh:
                return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
        except Exception:
            return float("nan")


def _load_bundle(dataset: str, cfg: dict, seed: int):
    if dataset == "waterbirds":
        from .data.waterbirds import load_waterbirds
        return load_waterbirds(cfg["dataset"], seed, split_spec=SplitSpec(), build_datasets=False)
    if dataset == "celeba":
        from .data.celeba import load_celeba
        return load_celeba(cfg["dataset"], seed, split_spec=SplitSpec(), build_datasets=False)
    raise ValueError(f"unknown dataset {dataset!r}")


def _features_for(backbone: str, paths_by_split: dict, y_by_split: dict, cfg: dict,
                  dataset: str) -> dict:
    if backbone == "clip_vitb32":
        clipcfg = cfg.get("clip", {})
        return {sp: l2_normalize(clip_features(
                    paths_by_split[sp], model_name=clipcfg.get("model_name", "ViT-B-32"),
                    pretrained=clipcfg.get("pretrained", "openai"),
                    device=clipcfg.get("device", "cuda"),
                    cache_dir=clipcfg.get("cache_dir", "results/cache_clip"),
                    tag=f"{dataset}_{sp}"))
                for sp in paths_by_split}
    if backbone == "resnet50_erm":
        rcfg = cfg.get("resnet", {})
        return resnet50_erm_features(paths_by_split, y_by_split, tag=dataset,
                                     device=rcfg.get("device", "cuda"),
                                     epochs=rcfg.get("epochs", 10), lr=rcfg.get("lr", 1e-3),
                                     batch_size=rcfg.get("batch_size", 128),
                                     max_train=rcfg.get("max_train"),
                                     cache_dir=rcfg.get("cache_dir", "results/cache_resnet"))
    if backbone in FROZEN_BACKBONES:
        fcfg = cfg.get("frozen", {})
        return {sp: frozen_features(paths_by_split[sp], name=backbone,
                                    device=fcfg.get("device", "cuda"),
                                    cache_dir=fcfg.get("cache_dir", "results/cache_frozen"),
                                    tag=f"{dataset}_{sp}",
                                    batch_size=fcfg.get("batch_size", 128),
                                    num_workers=fcfg.get("num_workers", 8))
                for sp in paths_by_split}
    raise ValueError(f"unknown backbone {backbone!r}; known: resnet50_erm, "
                     f"{', '.join(FROZEN_BACKBONES)}")


def build_griddata(dataset: str, backbone: str, cfg: dict, seed: int = 0) -> GridData:
    b = _load_bundle(dataset, cfg, seed)
    paths = {sp: b.meta["paths"][sp] for sp in ("train", "d_learn", "d_cal", "d_test")}
    y = {sp: np.asarray(b.y[sp]).astype(int) for sp in paths}
    grp = {sp: np.asarray(b.group_id[sp]).astype(int) for sp in paths}
    feats = _features_for(backbone, paths, y, cfg, dataset)

    eval_X = np.concatenate([feats["d_cal"], feats["d_test"]], axis=0)
    eval_y = np.concatenate([y["d_cal"], y["d_test"]])
    eval_g = np.concatenate([grp["d_cal"], grp["d_test"]])
    return GridData(backbone=backbone, dataset=dataset,
                    train=(feats["train"], y["train"], grp["train"]),
                    reweight=(feats["d_learn"], y["d_learn"], grp["d_learn"]),
                    eval_domain=(eval_X, eval_y, eval_g), n_classes=int(b.n_classes))
