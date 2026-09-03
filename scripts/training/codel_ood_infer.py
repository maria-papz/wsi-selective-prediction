"""
Scores TCGA codeletion (1p/19q) slides with the class-conditional density
(normalizing-flow) OOD models same logic as selective_prediction_bench_clean.ipynb -- same checkpoints (trained
on EBRAINS, dataset-independent), same three aggregations (topk/mean/
attention), same OOD_ENCODERS restriction (uni2, hoptimus only).

Direct port of ipd_brain_ood_infer.py, adapted for the codel cohort:
  - Attention comes from the codel ABMIL checkpoints
    (outputs/abmil_codel/<encoder>/fold<N>/best.ckpt), not the IDH ones --
    it needs to reflect what the codel classifier actually attends to.
  - Slide table is the codel-restricted TCGA cohort (merge of
    slide_table_tcga_{encoder}.csv against tcga_1p19q_labels.csv), not the
    full IDH population -- this cohort is already ground-truth-IDH-mutant
    by construction (see build_1p19q_labels.py), so no further filtering
    is needed here.

The flow checkpoints' two mutant classes (astrocytoma, oligodendroglioma)
map exactly onto codel's two classes, so no retraining is needed -- this
script only scores an already-trained, dataset-independent model against a
new (codel-restricted) population.

Usage:
    python scripts/training/codel_ood_infer.py --encoder uni2
    python scripts/training/codel_ood_infer.py --encoder hoptimus
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
FOLDS = [0, 1, 2, 3, 4]
CLASS_NAMES = ("gbm", "astrocytoma", "oligodendroglioma")
NUM_CLASSES = len(CLASS_NAMES)
OOD_AGGREGATIONS = ["topk", "attention", "mean"]


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
    from src.models.abmil import ABMIL
    return ABMIL(in_dim=in_dim, latent_dim=latent_dim, num_classes=n_classes, dropout=dropout)


def _abmil_checkpoint(encoder, fold):
    root = BASE / "outputs/abmil_codel" / encoder / f"fold{fold}"
    for pat in ("best.ckpt", "*best*.ckpt", "*.ckpt"):
        cands = sorted(root.glob(pat))
        if cands:
            return cands[-1]
    raise FileNotFoundError(f"No codel ABMIL checkpoint under {root}")


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

    res, n_top = {}, max(1, int(np.ceil(top_fraction * len(patch_ood))))
    res["topk"] = float(np.partition(patch_ood, len(patch_ood) - n_top)[-n_top:].mean())
    res["mean"] = float(patch_ood.mean())
    if attention is not None and len(attention) == len(patch_ood):
        w = np.clip(attention, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1 / len(w))
        res["attention"] = float(np.sum(w * patch_ood))
    else:
        res["attention"] = np.nan
    return res


def preload_features(slide, h5idx, bag_size, seed=42):
    """Read every slide's patch features from NFS exactly once -- see
    ipd_brain_ood_infer.py's own docstring for why this matters (avoids
    re-reading the same feature directory once per fold)."""
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


def load_codel_slide_table(encoder, dataset):
    slide = pd.read_csv(BASE / f"data/processed/slide_table_{dataset}_{encoder}.csv")
    slide["PATIENT"] = slide["PATIENT"].astype(str)
    if dataset == "tcga":
        labels = pd.read_csv(BASE / "data/raw/tcga/clinical/tcga_1p19q_labels.csv")
        labels["PATIENT"] = labels["PATIENT"].astype(str)
        slide = slide.merge(labels[["PATIENT", "codel_binary"]], on="PATIENT", how="inner")
    elif dataset == "ipd_brain":
        # Same merge convention as eval_codel_ipd_brain.py: slide table's PATIENT
        # matches the label CSV's case_id, not PATIENT.
        labels = pd.read_csv(BASE / "data/raw/ipd_brain/clinical/ipd_brain_1p19q_labels.csv")
        labels["case_id"] = labels["case_id"].astype(str)
        slide = slide.merge(labels[["case_id", "codel_binary", "ATRX"]],
                            left_on="PATIENT", right_on="case_id", how="inner")
    else:
        raise ValueError(dataset)
    return slide


def score_encoder(encoder, dataset="tcga", bag_size=5000, batch_size=1024, top_fraction=0.05):
    rootname, subdir = ENCODER_DIRS[encoder]
    feat_root = BASE / "data/processed/features" / dataset / rootname / subdir
    slide = load_codel_slide_table(encoder, dataset)

    print(f"Indexing {encoder} ({len(slide)} codel-eligible {dataset} slides) ...")
    h5idx = build_h5_index(feat_root)
    print(f"  preloading features (one NFS pass, bag_size={bag_size}) ...")
    feats_by_patient = preload_features(slide, h5idx, bag_size)
    print(f"  preloaded {len(feats_by_patient)}/{len(slide)} slides")
    attn_by_patient = build_attention_index(encoder, feats_by_patient)

    agg = {a: {} for a in OOD_AGGREGATIONS}
    for fold in FOLDS:
        print(f"  {encoder} density fold {fold}")
        model, dev = load_ood_model(encoder, fold)
        for pid, feats in feats_by_patient.items():
            # bag_size=0: features are already bagged once at preload time
            sc = score_slide_ood(model, dev, feats, batch_size, top_fraction,
                                 attention=attn_by_patient.get(pid), bag_size=0)
            if sc is None:
                continue
            for a in OOD_AGGREGATIONS:
                agg[a].setdefault(pid, []).append(sc[a])
        del model
        if _torch().cuda.is_available():
            _torch().cuda.empty_cache()

    pids = set().union(*[set(agg[a]) for a in OOD_AGGREGATIONS])
    rows = []
    for pid in pids:
        rec = {"patient": pid}
        for a in OOD_AGGREGATIONS:
            vals = [v for v in agg[a].get(pid, []) if np.isfinite(v)]
            rec[f"ood_{a}_{encoder}"] = float(np.mean(vals)) if vals else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    out_dir = BASE / f"outputs/ood/codel_{dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{encoder}_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out)} patients)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["uni2", "hoptimus"])
    ap.add_argument("--dataset", choices=["tcga", "ipd_brain"], default="tcga")
    ap.add_argument("--bag_size", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--top_fraction", type=float, default=0.05)
    args = ap.parse_args()
    score_encoder(args.encoder, dataset=args.dataset, bag_size=args.bag_size,
                  batch_size=args.batch_size, top_fraction=args.top_fraction)
