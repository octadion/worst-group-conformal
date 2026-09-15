"""Full-network fine-tuning of the backbone under three objectives.

    erm       plain cross-entropy
    groupdro  online GroupDRO over the four groups: per-batch group losses reweighted by
              exponentiated-gradient ascent on q
    reweight  group-balanced sampling, at inverse group frequency

Each objective keeps the recipe its own literature prescribes rather than one shared schedule.
GroupDRO on Waterbirds needs SGD with strong L2 to work at all, so running it on ERM's Adam recipe
would cripple the arm and turn that into a spurious null about representations; the manipulation
check in ``representation`` confirms the robust arms did become more robust before any null is
read. ``weight_decay`` defaults to 0, matching the ERM backbone in ``features``, where adding L2
inside Adam measurably lowered worst-group accuracy.

Returns the same ``{split: (N, d) L2-normalised array}`` dict as ``features``, so a fine-tuned
representation drops into ``griddata.build_griddata`` unchanged.

Needs a GPU; torch is imported inside the function, so the module stays importable without it.
Per-epoch checkpointing lets an interrupted run resume.
"""
from __future__ import annotations

import json
import os
from hashlib import sha1

import numpy as np

from .features import l2_normalize

OBJECTIVES = ("erm", "groupdro", "reweight")

# Host-RAM budget for decoded images sitting in DataLoader queues during feature extraction.
# The worker count is derived from this rather than inherited from training, because extraction
# runs at a larger batch and the product batch x workers x prefetch is what actually reserves RAM.
EXTRACT_INFLIGHT_BYTES = 1_500_000_000


def _paths_hash(paths) -> str:
    h = sha1()
    for p in paths:
        h.update(str(p).encode())
    return h.hexdigest()[:12]


def cache_key(objective: str, tag: str, paths_by_split: dict, *, epochs: int, seed: int,
              max_train, lr: float, batch_size: int, optimizer: str = "adam",
              weight_decay: float = 0.0, groupdro_eta: float = 0.01,
              init_weights: str = "IMAGENET1K_V2", amp: bool = True,
              select_by: str = "val_worst_group", val_frac: float = 0.1,
              val_min_per_group: int = 15) -> str:
    """Cache identity for one fine-tuned representation.

    EVERY knob that changes the learned representation must appear here. ``groupdro_eta``,
    ``optimizer`` and ``weight_decay`` are included precisely because tuning them is the expected
    response to an under-trained robust arm -- and a key that ignored them would answer the re-run
    with the old features and look like the change had no effect. ``groupdro_eta`` is folded in only
    for the objective it affects, so ERM/reweight caches stay valid across eta changes.
    """
    allp = [p for sp in sorted(paths_by_split) for p in paths_by_split[sp]]
    mt = "all" if max_train is None else int(max_train)
    extra = f"_eta{groupdro_eta:g}" if objective == "groupdro" else ""
    init = init_weights.replace("IMAGENET1K_", "in1k")     # the pretrained init IS the starting
    # Mixed precision changes the learned weights, so it belongs in the key. It is encoded only
    # when switched OFF, so the default (True) keeps the keys already on disk valid -- the same
    # trick used for ``groupdro_eta``, which appears only for the objective it affects.
    prec = "" if amp else "_fp32"
    # Which checkpoint is kept IS part of the representation, so the selection protocol is keyed.
    # Encoded only when selection is on, so the pre-selection caches keep their existing names and
    # remain available as a with/without ablation.
    sel = f"_sel{select_by}{val_frac:g}n{val_min_per_group}" if select_by else ""
    return (f"ft-{objective}_{tag}_{_paths_hash(allp)}_{init}_{epochs}ep_lr{lr:g}"  # representation
            f"_bs{batch_size}_wd{weight_decay:g}_{optimizer}{extra}{prec}{sel}"
            f"_mt{mt}_s{seed}").replace("/", "-")


def _save_checkpoint(torch, payload: dict, path: str, *, retries: int = 3) -> bool:
    """Write a checkpoint atomically, tolerating a flaky Google Drive FUSE mount.

    Colab's Drive mount intermittently fails mid-write on large files ("A Google Drive error has
    occurred"), which with a plain ``torch.save`` leaves a truncated file that then poisons the next
    resume. Writing to a sibling temp file and renaming means the destination is either the previous
    good checkpoint or the new one, never a half-written mix. A failure here is logged and skipped
    rather than raised: losing a checkpoint costs re-running some epochs, but aborting loses the
    whole run.
    """
    tmp = path + ".tmp"
    for attempt in range(1, retries + 1):
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)                    # atomic within a filesystem
            return True
        except Exception as e:
            print(f"[ckpt] write failed (attempt {attempt}/{retries}): {e}", flush=True)
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
    print(f"[ckpt] giving up on {path}; training continues without a resume point", flush=True)
    return False


def _group_weights(groups: np.ndarray) -> np.ndarray:
    """Per-sample sampling weight = 1 / (count of its group) -> group-balanced batches."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    return (1.0 / counts)[inv]


def finetune_features(paths_by_split: dict, y_by_split: dict, group_by_split: dict, *,
                      objective: str = "erm", tag: str = "waterbirds", device: str = "cuda",
                      epochs: int = 10, lr: float = 1e-3, weight_decay: float = 0.0,
                      batch_size: int = 128, image_size: int = 224, seed: int = 0,
                      max_train=None, groupdro_eta: float = 0.01,
                      optimizer: str = "adam", train_split: str = "train",
                      init_weights: str = "IMAGENET1K_V2",
                      cache_dir: str = "results/cache_finetune", ckpt_dir=None,
                      num_workers: int = 8, amp: bool = True, log_every: int = 1,
                      ckpt_every: int = 1, cache_dtype: str = "float32",
                      extract_batch_size=None, extract_inflight_bytes=None,
                      select_by: str = "val_worst_group", val_frac: float = 0.1,
                      val_min_per_group: int = 15) -> dict:
    """Fine-tune ResNet-50 end-to-end under ``objective`` and return penultimate features.

    Resumable: after every epoch a checkpoint (model/optimizer/scaler/GroupDRO ``q``/epoch) is
    written to ``ckpt_dir``; re-invoking with identical arguments continues from it. On completion
    the extracted features are cached to ``cache_dir`` and the checkpoint is no longer consulted.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {OBJECTIVES}")
    if select_by not in ("", "val_worst_group"):
        raise ValueError(f"unknown select_by {select_by!r}; use 'val_worst_group' or '' to disable")

    key = cache_key(objective, tag, paths_by_split, epochs=epochs, seed=seed,
                    max_train=max_train, lr=lr, batch_size=batch_size, optimizer=optimizer,
                    weight_decay=weight_decay, groupdro_eta=groupdro_eta,
                    init_weights=init_weights, amp=amp, select_by=select_by, val_frac=val_frac,
                    val_min_per_group=val_min_per_group)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, key + ".npz")
    if os.path.exists(cache_path):
        try:                                    # a Drive-truncated .npz must not pass as a hit
            z = np.load(cache_path)
            feats = {sp: np.asarray(z[sp], dtype=np.float32) for sp in z.files}
            if not feats or any(v.size == 0 for v in feats.values()):
                raise ValueError("cached arrays are empty")
            print(f"[ft-{objective} {tag} s{seed}] cache hit -> {cache_path}")
            return feats
        except Exception as e:
            print(f"[ft-{objective} {tag} s{seed}] cache unreadable ({e}); recomputing", flush=True)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import models, transforms
    from PIL import Image

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    # cudnn.benchmark and TF32 are both left OFF, deliberately, and this costs throughput.
    #
    # Neither is a hyperparameter, so neither belongs in ``cache_key`` -- which is exactly why both
    # are dangerous here: they change kernel selection (benchmark) or precision (TF32), so arms
    # computed under different settings are not numerically comparable, and nothing in the cache
    # would reveal it. The Waterbirds arms were all computed before either existed, so enabling
    # them for CelebA would leave the two datasets processed differently for a ~10% speedup.
    # Uniform processing across every arm is worth more than that.
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(image_size), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class _DS(Dataset):
        def __init__(self, paths, labels, groups=None):
            self.paths, self.labels = list(paths), np.asarray(labels)
            self.groups = np.zeros(len(self.paths), dtype=int) if groups is None else np.asarray(groups)

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            x = tf(Image.open(self.paths[i]).convert("RGB"))
            return x, int(self.labels[i]), int(self.groups[i])

    n_classes = int(len(np.unique(y_by_split[train_split])))
    # IMAGENET1K_V2 to match the backbone the submitted paper already used
    # (features.resnet50_erm_features). V1 is a materially weaker starting point (76.1% vs 80.9%
    # ImageNet top-1) and produced worst-group accuracies below the paper's published range, which
    # would leave the revision's tables contradicting the paper's own.
    net = models.resnet50(weights=getattr(models.ResNet50_Weights, init_weights))
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    net = net.to(dev, memory_format=torch.channels_last)

    # ---- training pool (optionally budget-subsampled; RANDOM, so in-domain correlation is kept)
    tr_paths = list(paths_by_split[train_split])
    tr_y = np.asarray(y_by_split[train_split]).astype(int)
    tr_g = np.asarray(group_by_split[train_split]).astype(int)
    if max_train is not None and len(tr_paths) > max_train:
        sub = np.random.default_rng(seed).choice(len(tr_paths), size=int(max_train), replace=False)
        tr_paths = [tr_paths[i] for i in sub]
        tr_y, tr_g = tr_y[sub], tr_g[sub]
        print(f"[ft-{objective} {tag} s{seed}] train subsample {len(tr_paths)}/"
              f"{len(paths_by_split[train_split])} (random: in-domain correlation preserved)")

    groups_sorted = np.array(sorted(np.unique(tr_g)))
    G = len(groups_sorted)
    gmap = {int(v): j for j, v in enumerate(groups_sorted)}
    tr_gk = np.array([gmap[int(v)] for v in tr_g])

    # ---- group-stratified validation slice, carved from TRAIN and never trained on.
    # Model selection on worst-group validation accuracy is the standard protocol in this
    # literature (Sagawa et al. 2020; Kirichenko et al. 2023; Liu et al. 2021) precisely because
    # these objectives overfit the minority group. Taking the last epoch instead let GroupDRO on
    # CelebA drive train worst-group accuracy to 0.978 while evaluation stayed at ERM's level.
    # The slice comes out of train rather than d_learn/d_cal/d_test so that DFR's reweighting
    # split and the conformal pools stay untouched -- selecting on d_learn would leak the
    # selection signal into the split DFR is later fitted on.
    va_idx = np.array([], dtype=int)
    if select_by and val_frac > 0:
        rng_v = np.random.default_rng(10_000 + seed)
        take = []
        for j in range(G):                       # stratified: the minority group is ~0.85% of
            idx = np.flatnonzero(tr_gk == j)     # CelebA, so a uniform draw could miss it
            # A floor, because val_frac alone is useless on a tiny group: Waterbirds' smallest
            # training group has ~56 examples, so 10% is 6 and the worst-group estimate has an SE
            # near 0.20. The floor is itself capped at a quarter of the group, so a small group
            # never loses most of its training examples to validation.
            k = max(int(round(val_frac * idx.size)), min(val_min_per_group, idx.size // 4))
            if idx.size >= 2 and k >= 1:
                take.append(rng_v.choice(idx, size=min(k, idx.size - 1), replace=False))
        if take:
            va_idx = np.sort(np.concatenate(take))
    keep = np.setdiff1d(np.arange(len(tr_paths)), va_idx, assume_unique=False)
    va_paths = [tr_paths[i] for i in va_idx]
    va_y, va_gk = tr_y[va_idx], tr_gk[va_idx]
    tr_paths = [tr_paths[i] for i in keep]
    tr_y, tr_gk = tr_y[keep], tr_gk[keep]
    if va_idx.size:
        per_g = [int((va_gk == j).sum()) for j in range(G)]
        se = 0.5 / max(1.0, float(min(per_g)) ** 0.5)     # worst case SE of the smallest group
        print(f"[ft-{objective} {tag} s{seed}] val slice {va_idx.size} (per group {per_g}); "
              f"training on {len(tr_paths)}; selection SE on the smallest group ~{se:.3f}"
              + ("  [NOISY: treat selection as cliff-avoidance, not fine tuning]"
                 if se > 0.08 else ""))

    ds = _DS(tr_paths, tr_y, tr_gk)
    dl_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                 drop_last=False, persistent_workers=num_workers > 0)
    if num_workers > 0:
        dl_kw["prefetch_factor"] = 4
    if objective == "reweight":
        w = _group_weights(tr_gk)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(ds), replacement=True)
        tr = DataLoader(ds, sampler=sampler, **dl_kw)
    else:
        tr = DataLoader(ds, shuffle=True, **dl_kw)

    if optimizer == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    use_amp = bool(amp) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit_none = nn.CrossEntropyLoss(reduction="none")

    q = np.ones(G) / G          # GroupDRO group weights (exponentiated-gradient ascent)
    start_ep = 0
    best = {"score": -1.0, "epoch": 0, "state": None}   # best-on-validation checkpoint

    def _val_worst_group():
        """Worst-group accuracy on the held-out slice. No grad, eval mode, restored after."""
        if not va_paths:
            return None
        vl = DataLoader(_DS(va_paths, va_y, va_gk), batch_size=int(extract_batch_size or batch_size),
                        shuffle=False, num_workers=min(4, num_workers), pin_memory=True)
        was_training = net.training
        net.eval()
        corr, seen_g = np.zeros(G), np.zeros(G)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for xb, yb, gb in vl:
                xb = xb.to(dev, non_blocking=True, memory_format=torch.channels_last)
                pred = net(xb).argmax(1).cpu().numpy()
                ok = (pred == yb.numpy()).astype(float)
                gnp = gb.numpy()
                for j in range(G):
                    m = gnp == j
                    if m.any():
                        corr[j] += ok[m].sum(); seen_g[j] += m.sum()
        del vl                                   # release its worker processes before training
        if was_training:
            net.train()
        acc = np.divide(corr, np.maximum(seen_g, 1))
        return float(acc[seen_g > 0].min()) if (seen_g > 0).any() else None

    # ---- resume
    ckpt_dir = ckpt_dir or os.path.join(cache_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, key + ".pt")
    if os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=dev)
            net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"]); q = np.asarray(ck["q"], dtype=np.float64)
            start_ep = int(ck["epoch"])
            if ck.get("best_state") is not None:
                best = {"score": float(ck["best_score"]), "epoch": int(ck["best_epoch"]),
                        "state": ck["best_state"]}
                print(f"[ft-{objective} {tag} s{seed}] restored best epoch {best['epoch']} "
                      f"(val worst-group {best['score']:.3f})")
            print(f"[ft-{objective} {tag} s{seed}] RESUMED from epoch {start_ep}/{epochs}")
        except Exception as e:                                   # corrupt/partial write -> restart
            print(f"[ft-{objective} {tag} s{seed}] checkpoint unreadable ({e}); starting fresh")
            start_ep = 0

    # ---- train
    import time as _time
    net.train()
    for ep in range(start_ep, epochs):
        run, seen, correct = 0.0, 0, 0
        g_correct, g_seen = np.zeros(G), np.zeros(G)
        _t0 = _time.time()
        for xb, yb, gb in tr:
            xb = xb.to(dev, non_blocking=True, memory_format=torch.channels_last)
            yb = yb.to(dev, non_blocking=True)
            gb_np = gb.numpy()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = net(xb)
                per_sample = crit_none(logits, yb)
                if objective == "groupdro":
                    # per-group mean loss -> q ascent -> q-weighted objective (Sagawa et al. 2020)
                    gl = torch.zeros(G, device=dev)
                    present = []
                    for j in range(G):
                        m = torch.as_tensor(gb_np == j, device=dev)
                        if m.any():
                            gl[j] = per_sample[m].mean()
                            present.append(j)
                    with torch.no_grad():
                        gl_np = gl.detach().float().cpu().numpy()
                    for j in present:
                        q[j] *= float(np.exp(groupdro_eta * gl_np[j]))
                    q = np.clip(q, 1e-12, None); q /= q.sum()
                    qt = torch.as_tensor(q, dtype=gl.dtype, device=dev)
                    loss = (qt * gl).sum()
                else:
                    loss = per_sample.mean()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

            with torch.no_grad():
                pred = logits.argmax(1)
                ok = (pred == yb).float().cpu().numpy()
            run += float(loss.detach()) * len(yb); seen += len(yb); correct += int(ok.sum())
            for j in range(G):
                m = gb_np == j
                if m.any():
                    g_correct[j] += float(ok[m].sum()); g_seen[j] += int(m.sum())

        if (ep + 1) % ckpt_every == 0 or (ep + 1) == epochs:
            _save_checkpoint(torch, {"model": net.state_dict(), "opt": opt.state_dict(),
                                     "scaler": scaler.state_dict(), "q": q.tolist(),
                                     "epoch": ep + 1,
                                     # without these a resumed run forgets the best epoch and
                                     # selects from the remaining ones only
                                     "best_score": best["score"], "best_epoch": best["epoch"],
                                     "best_state": best["state"]}, ckpt_path)
        # ---- model selection on held-out worst-group accuracy
        vwg = _val_worst_group()
        if vwg is not None and vwg > best["score"]:
            best = {"score": vwg, "epoch": ep + 1,
                    # .cpu() so the kept copy does not pin GPU memory for the rest of training
                    "state": {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}}

        if (ep + 1) % log_every == 0:
            gacc = np.divide(g_correct, np.maximum(g_seen, 1))
            wg = float(gacc[g_seen > 0].min()) if (g_seen > 0).any() else float("nan")
            qs = " q=[" + ",".join(f"{v:.2f}" for v in q) + "]" if objective == "groupdro" else ""
            vs = f" val-wg={vwg:.3f}{'*' if best['epoch'] == ep + 1 else ''}" if vwg is not None else ""
            el = max(1e-9, _time.time() - _t0)
            # img/s plus peak VRAM: if throughput is well under what the GPU can do and memory is
            # barely touched, the loader is starving it and num_workers is the knob to raise.
            mem = (f" vram={torch.cuda.max_memory_allocated()/2**30:.1f}G"
                   if dev.type == "cuda" else "")
            print(f"[ft-{objective} {tag} s{seed}] ep {ep+1}/{epochs} loss={run/max(1,seen):.4f} "
                  f"acc={correct/max(1,seen):.3f} train-wg={wg:.3f}{vs}{qs} "
                  f"[{seen/el:.0f} img/s, {el/60:.1f}m{mem}]", flush=True)

    # ---- restore the selected checkpoint. Without this the extracted features come from the
    # LAST epoch, which is exactly the checkpoint the selection exists to avoid.
    if best["state"] is not None:
        net.load_state_dict({k: v.to(dev) for k, v in best["state"].items()})
        print(f"[ft-{objective} {tag} s{seed}] selected epoch {best['epoch']}/{epochs} "
              f"(val worst-group {best['score']:.3f})", flush=True)
        best["state"] = None                      # free the CPU copy before extraction allocates
    elif select_by:
        print(f"[ft-{objective} {tag} s{seed}] no validation slice -- using the last epoch",
              flush=True)

    # ---- extract penultimate features for every split
    feat_net = torch.nn.Sequential(*list(net.children())[:-1]).to(dev).eval()
    # Extraction is a no-grad forward pass in eval mode, so BatchNorm uses running statistics and
    # the per-sample output is MATHEMATICALLY independent of batch composition -- which is why
    # ``extract_batch_size`` is not part of ``cache_key``. It is not bit-identical across batch
    # sizes though: with cudnn.benchmark the conv algorithm is selected per batch shape, so rounding
    # differs at ~1e-3 relative under AMP. That is orders below the seed-to-seed spread we report,
    # but arms cached at different extraction batches are not byte-comparable.
    ebs = int(extract_batch_size or batch_size)
    # A DataLoader holds batch_size x num_workers x prefetch_factor decoded images in host RAM.
    # At 224x224x3 float32 that is 0.57 MB each, so bs=512 with 16 workers reserves ~9 GB before a
    # single feature is written -- which is how a CelebA run exhausted RAM mid-extraction. Derive
    # the worker count from an explicit in-flight budget instead of inheriting the training value.
    img_bytes = 3 * image_size * image_size * 4
    budget = int(extract_inflight_bytes or EXTRACT_INFLIGHT_BYTES)
    ext_workers = max(2, min(num_workers, int(budget / (ebs * 2 * img_bytes))))
    print(f"[ft-{objective} {tag} s{seed}] extracting at bs={ebs}, workers={ext_workers} "
          f"(~{ebs * ext_workers * 2 * img_bytes / 2**30:.1f} GB in flight)", flush=True)
    out = {}
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for sp, paths in paths_by_split.items():
            dl = DataLoader(_DS(paths, y_by_split[sp]), batch_size=ebs, shuffle=False,
                            num_workers=ext_workers, pin_memory=True)
            n = len(paths)
            # Write straight into one preallocated array. Collecting chunks in a list and then
            # concatenating held three copies of the split at once (list + concat + normalised),
            # which is 3.7 GB for CelebA's train split alone.
            buf = None
            i = 0
            for xb, _, _ in dl:
                xb = xb.to(dev, non_blocking=True, memory_format=torch.channels_last)
                f = feat_net(xb).squeeze(-1).squeeze(-1).float().cpu().numpy()
                if buf is None:
                    buf = np.empty((n, f.shape[1]), dtype=np.float32)
                buf[i:i + f.shape[0]] = f
                i += f.shape[0]
            assert buf is not None and i == n, f"extracted {i} of {n} rows for split {sp!r}"
            nrm = np.linalg.norm(buf, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            buf /= nrm                                    # in place: no second full-size copy
            out[sp] = buf
            print(f"[ft-{objective} {tag} s{seed}] extracted {sp}: {buf.shape}", flush=True)

    # Same atomic-write discipline for the payload that actually matters: a truncated .npz would
    # register as a cache hit on the next run and silently feed corrupt features into the analysis.
    # ``cache_dtype="float16"`` halves the stored size, which matters on CelebA: at 2048-d and
    # ~192k samples a run is 1.6 GB in float32, and nine runs would exhaust a 15 GB Drive on their
    # own. Features are L2-normalised (values ~0.02), so float16's ~1e-3 relative precision is well
    # below the noise the downstream linear head and conformal quantiles already carry. Arrays are
    # always returned in float32 regardless, so only the on-disk copy is reduced.
    to_store = {k: v.astype(cache_dtype, copy=False) for k, v in out.items()}
    tmp_npz = cache_path + ".tmp.npz"
    np.savez(tmp_npz, **to_store)
    os.replace(tmp_npz, cache_path)
    with open(os.path.join(cache_dir, key + ".meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"objective": objective, "tag": tag, "epochs": epochs, "lr": lr,
                   "weight_decay": weight_decay, "batch_size": batch_size, "seed": seed,
                   "max_train": max_train, "optimizer": optimizer,
                   "init_weights": init_weights, "amp": bool(amp),
                   "select_by": select_by, "val_frac": val_frac,
                   "val_min_per_group": val_min_per_group,
                   "selected_epoch": best["epoch"], "selected_val_wg": best["score"],
                   "groupdro_eta": groupdro_eta if objective == "groupdro" else None,
                   "n_train": len(tr_paths), "groups": groups_sorted.tolist()}, fh, indent=2)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)                                     # finished: free the checkpoint
    print(f"[ft-{objective} {tag} s{seed}] cached -> {cache_path}")
    return out
