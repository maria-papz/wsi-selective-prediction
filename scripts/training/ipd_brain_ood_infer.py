"""
Scores IPD Brain slides with the class-conditional density (normalizing-flow)
OOD models

Also computes the energy score (mean/topk aggregations of
`-logsumexp(class_log_probs + class_log_prior)`, matching
`energy_ood_mean`/`energy_ood_top5_mean` in
outputs/ood_evaluation_analysis/merged_case_scores.csv for TCGA/EBRAINS)


Usage:
    python scripts/training/ipd_brain_ood_infer.py --encoder uni2
    python scripts/training/ipd_brain_ood_infer.py --encoder hoptimus
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent

ENCODER_DIRS = {
    "uni2":     ("uni",      "uni2-49b04e14"),
    "conch":    ("conch",    "conch-49b04e14"),
    "hoptimus": ("hoptimus", "h-optimus-1-49b04e14"),
}
ABMIL_DIRS = {"uni2": "uni2", "conch": "conch", "hoptimus": "hoptimus"}
FOLDS = [0, 1, 2, 3, 4]
CLASS_NAMES = ("gbm", "astrocytoma", "oligodendroglioma")
NUM_CLASSES = len(CLASS_NAMES)
OOD_AGGREGATIONS = ["topk", "attention", "mean"]
ENERGY_AGGREGATIONS = ["topk", "mean"]   # no attention-weighted energy -- matches
                                          # merged_case_scores.csv, which only has
                                          # energy_ood_mean/energy_ood_p95, no
                                          # attention-weighted energy variant either


def _torch():
    import torch
    return torch


def get_device():
    torch = _torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_checkpoint(encoder, fold):
    root = BASE / "outputs/class_conditional_density" / encoder / f"fold{fold}"
    cands = sorted(root.glob("*/best.ckpt"))
    if not cands:
        raise FileNotFoundError(f"No density checkpoint under {root}")
    return cands[-1]


def load_ood_model(encoder, fold):
    torch = _torch()
    from src.models.natpn import ClassConditionalPatchDensityModel, OrthogonalProjection
    from src.models.maf import MaskedAutoregressiveFlow
    from src.models.radial_flow import RadialFlow
    dev = get_device()
    ckpt = torch.load(find_checkpoint(encoder, fold), map_location=dev, weights_only=False)
    in_dim, latent = int(ckpt["in_dim"]), int(ckpt["latent_dim"])
    ftype, flayers = str(ckpt["args"]["flow_type"]), int(ckpt["args"]["flow_layers"])
    proj = OrthogonalProjection(in_dim=in_dim, latent_dim=latent, trainable=False)
    if ftype == "maf":
        flows = [MaskedAutoregressiveFlow(dim=latent, num_layers=flayers,
                 use_batch_norm=False) for _ in range(NUM_CLASSES)]
    elif ftype == "radial":
        flows = [RadialFlow(dim=latent, num_layers=flayers) for _ in range(NUM_CLASSES)]
    else:
        raise ValueError(ftype)
    model = ClassConditionalPatchDensityModel(
        encoder=proj, flows=flows, latent_dim=latent,
        num_classes=NUM_CLASSES, class_names=CLASS_NAMES).to(dev)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    return model, dev


def build_abmil_model(in_dim, latent_dim=256, n_classes=2, dropout=0.25):
    torch = _torch(); import torch.nn as nn
    try:
        from src.models.abmil import ABMIL
        return ABMIL(in_dim=in_dim, latent_dim=latent_dim, num_classes=n_classes, dropout=dropout)
    except Exception as e:
        print(f"    [abmil] could not import src.models.abmil ({e}); using verified fallback")

    class GatedAttention(nn.Module):
        def __init__(self, L, D):
            super().__init__()
            self.V = nn.Linear(L, D); self.U = nn.Linear(L, D); self.w = nn.Linear(D, 1)

        def forward(self, h):
            attn = torch.softmax(self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))), dim=0)
            return (attn * h).sum(0), attn.squeeze(-1)

    class ABMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout))
            self.attention = GatedAttention(latent_dim, latent_dim)
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(latent_dim, n_classes))

        def forward(self, h, return_attention=False):
            h = self.proj(h); z, attn = self.attention(h); logits = self.classifier(z)
            return (logits, z, attn) if return_attention else (logits, z)

    return ABMIL()


def _abmil_checkpoint(encoder, fold):
    root = BASE / "outputs/abmil" / ABMIL_DIRS[encoder] / f"fold{fold}"
    for pat in ("best.ckpt", "*best*.ckpt", "*.ckpt", "*best*.pt", "*.pt"):
        cands = sorted(root.glob(pat))
        if cands:
            return cands[-1]
    raise FileNotFoundError(f"No ABMIL checkpoint under {root}")


def load_abmil_model(encoder, fold, in_dim):
    torch = _torch(); dev = get_device()
    ckpt = torch.load(_abmil_checkpoint(encoder, fold), map_location=dev, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    latent = int(args.get("latent_dim", 256)) if isinstance(args, dict) else 256
    drop = float(args.get("dropout", 0.25)) if isinstance(args, dict) else 0.25
    model = build_abmil_model(in_dim, latent_dim=latent, dropout=drop).to(dev)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, dev


def extract_attention(model, feats):
    torch = _torch(); dev = get_device()
    x = torch.from_numpy(feats).float().to(dev)
    with torch.no_grad():
        try:
            out = model(x, return_attention=True)
        except TypeError:
            out = model(x)
    a = out[-1] if isinstance(out, (tuple, list)) else out
    a = np.asarray(a.detach().cpu().numpy()).ravel()
    if len(a) != len(feats) or not np.isfinite(a).all():
        raise RuntimeError(f"Attention length {len(a)} != patches {len(feats)} or non-finite.")
    return a


def build_h5_index(root: Path):
    if not root.exists():
        raise FileNotFoundError(root)
    idx = {}
    for p in root.rglob("*.h5"):
        idx.setdefault(p.name, p); idx.setdefault(p.stem, p)
    return idx


def resolve_h5_path(row, index):
    keys = []
    for col in ("FILENAME", "filename", "file_name"):
        if col in row.index and pd.notna(row[col]):
            name = Path(str(row[col])).name
            keys += [name, Path(name).stem]
    patient = str(row["PATIENT"]); keys += [f"{patient}.h5", patient]
    for k in keys:
        if k in index:
            return index[k]
    return None


def read_feats(h5_path):
    import h5py
    if h5_path is None or not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as f:
        if "feats" not in f:
            return None
        feats = f["feats"][:]
    return feats if len(feats) else None


def score_slide_ood(model, dev, feats, batch_size, top_fraction, attention=None, bag_size=5000, seed=42):
    torch = _torch()
    if feats is None:
        return None
    if bag_size > 0 and len(feats) > bag_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(feats), bag_size, replace=False)
        feats = feats[idx]
        attention = (attention[idx] if (attention is not None and
                     len(attention) >= idx.max() + 1) else None)
    chunks = []
    with torch.no_grad():
        for s in range(0, len(feats), batch_size):
            b = torch.from_numpy(feats[s:s + batch_size]).float().to(dev)
            chunks.append(model(b).detach().cpu().numpy())
    class_log_probs = np.concatenate(chunks, 0)   # (n_patches, NUM_CLASSES)
    patch_ood = -class_log_probs.max(axis=1)

    log_prior = model.class_log_prior.detach().cpu().numpy()
    joint = class_log_probs + log_prior[None, :]
    joint_max = joint.max(axis=1, keepdims=True)
    patch_energy = -(joint_max.squeeze(1) + np.log(np.exp(joint - joint_max).sum(axis=1)))

    res, n_top = {}, max(1, int(np.ceil(top_fraction * len(patch_ood))))
    res["topk"] = float(np.partition(patch_ood, len(patch_ood) - n_top)[-n_top:].mean())
    res["mean"] = float(patch_ood.mean())
    res["energy_topk"] = float(np.partition(patch_energy, len(patch_energy) - n_top)[-n_top:].mean())
    res["energy_mean"] = float(patch_energy.mean())
    if attention is not None and len(attention) == len(patch_ood):
        w = np.clip(attention, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1 / len(w))
        res["attention"] = float(np.sum(w * patch_ood))
    else:
        res["attention"] = np.nan
    return res


def preload_features(slide, h5idx, bag_size, seed=42):
    feats_by_patient = {}
    rng = np.random.default_rng(seed)
    for _, row in slide.iterrows():
        pid = str(row["PATIENT"])
        feats = read_feats(resolve_h5_path(row, h5idx))
        if feats is None:
            continue
        if bag_size > 0 and len(feats) > bag_size:
            idx = rng.choice(len(feats), bag_size, replace=False)
            feats = feats[idx]
        feats_by_patient[pid] = feats
    return feats_by_patient


def build_attention_index(encoder, feats_by_patient):
    in_dim = next(iter(feats_by_patient.values())).shape[1] if feats_by_patient else None
    if in_dim is None:
        print(f"  [attention {encoder}] no features found; skipping"); return {}
    models = [load_abmil_model(encoder, f, in_dim)[0] for f in FOLDS]
    attn = {}
    for pid, feats in feats_by_patient.items():
        try:
            attn[pid] = np.mean([extract_attention(m, feats) for m in models], axis=0)
        except Exception as e:
            print(f"  [attention {encoder} {pid}] {e}")
    if _torch().cuda.is_available():
        _torch().cuda.empty_cache()
    return attn


def score_encoder(encoder, bag_size=5000, batch_size=1024, top_fraction=0.05):
    rootname, subdir = ENCODER_DIRS[encoder]
    feat_root = BASE / "data/processed/features/ipd_brain" / rootname / subdir
    slide = pd.read_csv(BASE / f"data/processed/slide_table_ipd_brain_{encoder}.csv")
    slide["PATIENT"] = slide["PATIENT"].astype(str)

    print(f"Indexing {encoder} ({len(slide)} slides) ...")
    h5idx = build_h5_index(feat_root)
    print(f"  preloading features (one NFS pass, bag_size={bag_size}) ...")
    feats_by_patient = preload_features(slide, h5idx, bag_size)
    print(f"  preloaded {len(feats_by_patient)}/{len(slide)} slides")
    attn_by_patient = build_attention_index(encoder, feats_by_patient)

    all_keys = OOD_AGGREGATIONS + [f"energy_{a}" for a in ENERGY_AGGREGATIONS]
    agg = {a: {} for a in all_keys}
    for fold in FOLDS:
        print(f"  {encoder} density fold {fold}")
        model, dev = load_ood_model(encoder, fold)
        for pid, feats in feats_by_patient.items():
            # bag_size=0 here: features are already bagged once at preload
            # time, so score_slide_ood must not re-bag (which would draw a
            # different sub-sample each fold using the default seed).
            sc = score_slide_ood(model, dev, feats, batch_size, top_fraction,
                                 attention=attn_by_patient.get(pid), bag_size=0)
            if sc is None:
                continue
            for a in all_keys:
                agg[a].setdefault(pid, []).append(sc[a])
        del model
        if _torch().cuda.is_available():
            _torch().cuda.empty_cache()

    pids = set().union(*[set(agg[a]) for a in all_keys])
    rows = []
    for pid in pids:
        rec = {"patient": pid}
        for a in all_keys:
            vals = [v for v in agg[a].get(pid, []) if np.isfinite(v)]
            rec[f"ood_{a}_{encoder}"] = float(np.mean(vals)) if vals else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    out_dir = BASE / "outputs/ood/ipd_brain"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{encoder}_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out)} patients)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(ENCODER_DIRS))
    ap.add_argument("--bag_size", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--top_fraction", type=float, default=0.05)
    args = ap.parse_args()
    score_encoder(args.encoder, bag_size=args.bag_size, batch_size=args.batch_size,
                  top_fraction=args.top_fraction)
