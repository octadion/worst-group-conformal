"""Frozen backbone features, cached to disk and L2-normalised.

    resnet50_erm   ResNet-50 trained in-domain with ERM, then its penultimate 2048-d features
    clip_vitb32    frozen CLIP ViT-B/32, 512-d, image-text contrastive
    dinov2_vitb14  frozen DINOv2 ViT-B/14, 768-d, self-supervised
    vit_b16_in1k   frozen supervised ImageNet ViT-B/16, 768-d

The four span two architecture families and three pretraining regimes; each keeps its own
canonical preprocessing, since DINOv2 and CLIP do not share ImageNet statistics.

Every extractor checks its cache before importing torch, so a re-run over cached features needs
neither a GPU nor torch. Cache keys hash the exact path list, so a changed split misses.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

__all__ = ["l2_normalize", "clip_features", "resnet50_erm_features", "frozen_features",
           "FROZEN_BACKBONES"]

FROZEN_BACKBONES = ("clip_vitb32", "dinov2_vitb14", "vit_b16_in1k")


def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def _paths_hash(paths: list) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(os.path.basename(p).encode("utf-8", "ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def clip_features(paths: list, *, model_name: str = "ViT-B-32", pretrained: str = "openai",
                  device: str = "cuda", cache_dir: str = "results/cache_clip",
                  batch_size: int = 64, tag: str = "") -> np.ndarray:
    """(N, d) L2-normalised CLIP image embeddings."""
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{model_name}_{pretrained}_{tag}_{_paths_hash(paths)}_{len(paths)}".replace("/", "-")
    cache_path = os.path.join(cache_dir, f"clipfeat_{key}.npy")
    if os.path.exists(cache_path):
        feats = np.load(cache_path)
        if feats.shape[0] == len(paths):
            return feats

    try:
        import open_clip
        import torch
        from PIL import Image
    except Exception as e:
        raise RuntimeError(f"CLIP feature extraction needs open_clip, torch and PIL: {e}")

    model, _, preprocess = open_clip.create_model_and_transforms(model_name,
                                                                pretrained=pretrained,
                                                                device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    feats = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch])
            emb = model.encode_image(imgs.to(device))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.float().cpu().numpy())
    out = np.concatenate(feats, axis=0)
    np.save(cache_path, out)
    return out


def resnet50_erm_features(paths_by_split: dict, y_by_split: dict, *, train_split="train",
                          cache_dir="results/cache_resnet", tag="", device="cuda", epochs=10,
                          lr=1e-3, batch_size=128, image_size=224, seed=0,
                          max_train=None) -> dict:
    """Train ResNet-50 with plain cross-entropy on ``train_split``, then extract every split.

    ``max_train`` subsamples the backbone's training set at random rather than class-balanced,
    which preserves the composited spurious correlation the ERM backbone is supposed to learn.
    Extraction still covers every split in full.
    """
    os.makedirs(cache_dir, exist_ok=True)
    all_paths = [p for sp in paths_by_split for p in paths_by_split[sp]]
    key = f"resnet50erm_{tag}_{_paths_hash(all_paths)}_{epochs}ep_{seed}".replace("/", "-")
    cache_path = os.path.join(cache_dir, f"{key}.npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return {sp: z[sp] for sp in paths_by_split}

    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
        from torchvision.models import ResNet50_Weights
    except Exception as e:
        raise RuntimeError(f"resnet50_erm_features needs torch, torchvision and PIL: {e}")

    weights = ResNet50_Weights.IMAGENET1K_V2
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(image_size),
        transforms.ToTensor(), weights.transforms().__class__().normalize
        if hasattr(weights.transforms(), "normalize") else transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    class _DS(Dataset):
        def __init__(self, paths, labels):
            self.paths, self.labels = paths, labels

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return tf(Image.open(self.paths[i]).convert("RGB")), int(self.labels[i])

    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    n_classes = int(len(np.unique(y_by_split[train_split])))
    net = models.resnet50(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    net = net.to(dev)

    tr_paths = list(paths_by_split[train_split])
    tr_y = np.asarray(y_by_split[train_split])
    if max_train is not None and len(tr_paths) > max_train:
        sub = np.random.default_rng(seed).choice(len(tr_paths), size=int(max_train), replace=False)
        tr_paths = [tr_paths[i] for i in sub]
        tr_y = tr_y[sub]
        print(f"[resnet50-erm {tag}] training on random subsample {len(tr_paths)}/"
              f"{len(paths_by_split[train_split])}")
    tr = DataLoader(_DS(tr_paths, tr_y), batch_size=batch_size, shuffle=True, num_workers=2)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    net.train()
    for ep in range(epochs):
        run = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()
            run += float(loss)
        print(f"[resnet50-erm {tag}] epoch {ep + 1}/{epochs} loss={run / max(1, len(tr)):.4f}")

    feat_net = nn.Sequential(*list(net.children())[:-1]).to(dev).eval()
    out = {}
    with torch.no_grad():
        for sp, paths in paths_by_split.items():
            dl = DataLoader(_DS(paths, y_by_split[sp]), batch_size=batch_size, shuffle=False,
                            num_workers=2)
            feats = []
            for xb, _ in dl:
                feats.append(feat_net(xb.to(dev)).squeeze(-1).squeeze(-1).cpu().numpy())
            out[sp] = l2_normalize(np.concatenate(feats, axis=0))
    np.savez(cache_path, **out)
    return out


def _build_frozen(name: str, device):
    """(model, transform, dim) for one frozen backbone, in its own canonical preprocessing."""
    if name not in FROZEN_BACKBONES or name == "clip_vitb32":
        raise ValueError(f"unknown frozen backbone {name!r}; "
                         f"choose from {[b for b in FROZEN_BACKBONES if b != 'clip_vitb32']} "
                         f"(clip_vitb32 has its own loader)")
    import torch
    from torchvision import transforms

    if name == "vit_b16_in1k":
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        w = ViT_B_16_Weights.IMAGENET1K_V1
        net = vit_b_16(weights=w)
        net.heads = torch.nn.Identity()
        return net.to(device).eval(), w.transforms(), 768

    if name == "dinov2_vitb14":
        net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        tf = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return net.to(device).eval(), tf, 768

    raise AssertionError(f"unreachable: {name!r}")


def frozen_features(paths: list, *, name: str, device: str = "cuda",
                    cache_dir: str = "results/cache_frozen", tag: str = "",
                    batch_size: int = 128, num_workers: int = 8,
                    inflight_bytes: int = 1_500_000_000) -> np.ndarray:
    """Worker count follows a host-RAM budget: a loader holds batch x workers x prefetch images."""
    if name not in FROZEN_BACKBONES or name == "clip_vitb32":
        raise ValueError(f"unknown frozen backbone {name!r}")
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{name}_{tag}_{_paths_hash(list(paths))}".replace("/", "-")
    cache_path = os.path.join(cache_dir, f"frozen_{key}.npy")
    if os.path.exists(cache_path):
        feats = np.load(cache_path)
        if feats.shape[0] == len(paths):
            return feats
        print(f"[{name} {tag}] cached {feats.shape[0]} rows but {len(paths)} paths; recomputing")

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    net, tf, dim = _build_frozen(name, dev)

    class _DS(Dataset):
        def __init__(self, p):
            self.p = list(p)

        def __len__(self):
            return len(self.p)

        def __getitem__(self, i):
            return tf(Image.open(self.p[i]).convert("RGB"))

    img_bytes = 3 * 224 * 224 * 4
    nw = max(2, min(num_workers, int(inflight_bytes / (batch_size * 2 * img_bytes))))
    dl = DataLoader(_DS(paths), batch_size=batch_size, shuffle=False, num_workers=nw,
                    pin_memory=True)
    print(f"[{name} {tag}] extracting {len(paths)} images, bs={batch_size}, workers={nw}",
          flush=True)

    out, i = None, 0
    with torch.no_grad():
        for xb in dl:
            f = net(xb.to(dev, non_blocking=True)).float().cpu().numpy()
            if out is None:
                out = np.empty((len(paths), f.shape[1]), dtype=np.float32)
            out[i:i + f.shape[0]] = f
            i += f.shape[0]
    assert out is not None and i == len(paths), f"extracted {i} of {len(paths)}"
    out = l2_normalize(out)
    tmp = cache_path + ".tmp.npy"
    np.save(tmp, out)
    os.replace(tmp, cache_path)
    print(f"[{name} {tag}] cached {out.shape} -> {cache_path}", flush=True)
    return out
