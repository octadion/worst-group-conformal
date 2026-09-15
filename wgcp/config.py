"""Where the data and feature caches live, and how one (backbone, dataset) cell is built.

Dataset roots come from ``WATERBIRDS_ROOT`` and ``CELEBA_ROOT``; feature caches default to
``caches/`` beside the repository. Waterbirds can download itself, CelebA cannot.
"""
from __future__ import annotations

import os

from .griddata import build_griddata

DATASETS = ("waterbirds", "celeba")
BACKBONES = ("resnet50_erm", "clip_vitb32", "dinov2_vitb14", "vit_b16_in1k")

WATERBIRDS_URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
# The ResNet-50 backbone is trained on a random 30,000-image subsample of CelebA's training split.
# It is part of the cache key: changing it re-extracts rather than silently reusing.
CELEBA_RESNET_MAX_TRAIN = 30_000
DEFAULT_CACHE = os.environ.get("WGCP_CACHE", "caches")


# Fine-tuning recipes. Each objective keeps the schedule its own literature prescribes, and the
# last epoch is used rather than a selected checkpoint, so SELECT_BY is empty.
FT_RECIPE = {
    "waterbirds": {
        "erm": {"optimizer": "adam", "lr": 1e-3, "weight_decay": 0.0, "epochs": 10},
        "reweight": {"optimizer": "adam", "lr": 1e-3, "weight_decay": 0.0, "epochs": 10},
        "groupdro": {"optimizer": "sgd", "lr": 1e-3, "weight_decay": 1e-2, "epochs": 20,
                     "groupdro_eta": 0.05},
    },
    "celeba": {
        "erm": {"optimizer": "adam", "lr": 1e-3, "weight_decay": 0.0, "epochs": 6},
        "reweight": {"optimizer": "adam", "lr": 1e-3, "weight_decay": 0.0, "epochs": 6},
        "groupdro": {"optimizer": "sgd", "lr": 1e-3, "weight_decay": 1e-2, "epochs": 6,
                     "groupdro_eta": 0.05},
    },
}
SELECT_BY = ""
FT_SEEDS = {"waterbirds": (0, 1, 2, 3, 4), "celeba": (0, 1, 2)}


def dataset_root(dataset: str) -> str:
    var = {"waterbirds": "WATERBIRDS_ROOT", "celeba": "CELEBA_ROOT"}[dataset]
    root = os.environ.get(var)
    if not root:
        raise RuntimeError(f"set {var} to the extracted {dataset} directory "
                           f"(scripts/prepare_data.py does this for Waterbirds)")
    return root


def cfg_for(dataset: str, *, device: str = "cuda", cache_dir: str = DEFAULT_CACHE,
            download: bool = False) -> dict:
    cfg = {
        "clip": {"model_name": "ViT-B-32", "pretrained": "openai", "device": device,
                 "cache_dir": os.path.join(cache_dir, "clip")},
        "resnet": {"device": device, "epochs": 10, "lr": 1e-3, "batch_size": 128,
                   "cache_dir": os.path.join(cache_dir, "resnet")},
        "frozen": {"device": device, "cache_dir": os.path.join(cache_dir, "frozen"),
                   "batch_size": 128, "num_workers": 4},
    }
    if dataset == "waterbirds":
        cfg["dataset"] = {"root": dataset_root("waterbirds"), "image_size": 224, "n_classes": 2,
                          "download": download, "url": WATERBIRDS_URL}
    else:
        cfg["dataset"] = {"root": dataset_root("celeba"), "n_classes": 2}
        cfg["resnet"]["max_train"] = CELEBA_RESNET_MAX_TRAIN
    return cfg


def build_cell(backbone: str, dataset: str, *, device: str = "cuda",
               cache_dir: str = DEFAULT_CACHE, seed: int = 0):
    """One GridData. A warm cache makes this seconds of disk read and needs no GPU."""
    return build_griddata(dataset, backbone, cfg_for(dataset, device=device, cache_dir=cache_dir),
                          seed=seed)
