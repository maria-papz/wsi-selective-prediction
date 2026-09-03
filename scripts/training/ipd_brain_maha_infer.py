"""
Scores IPD Brain slides with the closed-form Gaussian (Mahalanobis-distance)
OOD ablation

Usage:
    python scripts/training/ipd_brain_maha_infer.py --encoder uni2
    python scripts/training/ipd_brain_maha_infer.py --encoder hoptimus
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ipd_brain_ood_infer import (  # noqa: E402
    BASE, ENCODER_DIRS, FOLDS,
    _torch, build_attention_index, build_h5_index, find_checkpoint,
    load_ood_model, preload_features,
)

OOD_ENCODERS = ["uni2", "hoptimus"]  # matches the main bench: no OOD for conch
OOD_AGGREGATIONS = ["topk", "attention", "mean"]
TOP_FRACTION, SHARED_COV = 0.05, True


def _ckpt(encoder, fold):
    torch = _torch()
    return torch.load(find_checkpoint(encoder, fold), map_location="cpu", weights_only=False)


def to_density_class(s):
    s = str(s).strip().lower()
    if s == "gbm" or "glioblastoma" in s:
        return "gbm"
    if "oligodendroglioma" in s:
        return "oligodendroglioma"
    if "astrocytoma" in s:
        return "astrocytoma"
    return None


def load_subtype_by_patient():
    eb = pd.read_csv(BASE / "data/raw/ebrains/clinical/ebrains_idh_cases.csv")
    eb["uuid"] = eb["uuid"].astype(str)
    return {
        r["uuid"]: to_density_class(r["tumour_type"])
        for _, r in eb.iterrows()
        if to_density_class(r["tumour_type"]) is not None
    }


def get_flow_projection(encoder, fold):
    """The frozen 64-d projection from a fold's density checkpoint, so
    Mahalanobis scores live in the same latent the flow scores in."""
    model, dev = load_ood_model(encoder, fold)
    proj = model.encoder
    torch = _torch()

    def project(feats):
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(feats)).float().to(dev)
            return proj(x).cpu().numpy()
    return project


def read_feats(h5_path):
    import h5py
    if h5_path is None or not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as f:
        if "feats" not in f:
            return None
        feats = f["feats"][:]
    return feats if len(feats) else None


def fit_class_gaussians(feats, labels, shared_cov=True, eps=1e-6):
    classes = np.unique(labels)
    means = {c: feats[labels == c].mean(0) for c in classes}
    d = feats.shape[1]
    if shared_cov:
        pooled = np.zeros((d, d))
        for c in classes:
            r = feats[labels == c] - means[c]
            pooled += r.T @ r
        prec = np.linalg.inv(pooled / len(feats) + eps * np.eye(d))
        precisions = {c: prec for c in classes}
    else:
        precisions = {}
        for c in classes:
            r = feats[labels == c] - means[c]
            precisions[c] = np.linalg.inv((r.T @ r) / len(r) + eps * np.eye(d))
    return means, precisions


def maha_patch_score(feats, means, precisions):
    dists = [np.einsum("ij,jk,ik->i", feats - means[c], precisions[c], feats - means[c])
             for c in means]
    return np.min(np.stack(dists, 1), axis=1)  # nearest class; higher = more OOD


def preload_ebrains_reference(encoder, folds, subtype_by_patient):
    ckpts = [_ckpt(encoder, fold) for fold in folds]
    feature_dir = ckpts[0]["feature_dir"]
    idx = build_h5_index(Path(feature_dir))
    all_patients = set()
    for c in ckpts:
        all_patients.update(map(str, c["train_patients"]))
    feats_by_patient = {}
    for pid in all_patients:
        if subtype_by_patient.get(pid) is None:
            continue
        h5 = idx.get(f"{pid}.h5") or idx.get(pid)
        f = read_feats(h5)
        if f is not None:
            feats_by_patient[pid] = f
    print(f"    [ref {encoder}] preloaded {len(feats_by_patient)}/{len(all_patients)} "
          f"union-of-folds reference patients")
    return ckpts, feats_by_patient


def build_fold_reference(ckpt, encoder, fold, subtype_by_patient, feats_by_patient):
    name_to_class = {n.lower(): i for i, n in enumerate(ckpt["class_names"])}
    F, L, miss = [], [], 0
    for pid in map(str, ckpt["train_patients"]):
        sub = subtype_by_patient.get(pid)
        f = feats_by_patient.get(pid)
        if sub is None or sub not in name_to_class or f is None:
            miss += 1
            continue
        F.append(f)
        L.append(np.full(len(f), name_to_class[sub], int))
    if miss:
        print(f"    [ref {encoder} f{fold}] dropped {miss} patients")
    return np.concatenate(F), np.concatenate(L)


def aggregate_maha(patch_maha, attention, top_fraction=TOP_FRACTION):
    res, n_top = {}, max(1, int(np.ceil(top_fraction * len(patch_maha))))
    res["topk"] = float(np.partition(patch_maha, len(patch_maha) - n_top)[-n_top:].mean())
    res["mean"] = float(patch_maha.mean())
    if attention is not None and len(attention) == len(patch_maha):
        w = np.clip(attention, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1 / len(w))
        res["attention"] = float(np.sum(w * patch_maha))
    else:
        res["attention"] = np.nan
    return res


def score_encoder(encoder, bag_size=5000):
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

    subtype_by_patient = load_subtype_by_patient()
    print("subtype counts:", pd.Series(list(subtype_by_patient.values())).value_counts().to_dict())

    ckpts, ref_feats_by_patient = preload_ebrains_reference(encoder, FOLDS, subtype_by_patient)

    fold_scores = {a: {} for a in OOD_AGGREGATIONS}
    for fold, ckpt in zip(FOLDS, ckpts):
        print(f"  maha {encoder} fold {fold}")
        project = get_flow_projection(encoder, fold)
        rf, rl = build_fold_reference(ckpt, encoder, fold, subtype_by_patient, ref_feats_by_patient)
        means, precisions = fit_class_gaussians(project(rf), rl, shared_cov=SHARED_COV)
        for pid, feats in feats_by_patient.items():
            patch_maha = maha_patch_score(project(feats), means, precisions)
            for k, v in aggregate_maha(patch_maha, attn_by_patient.get(pid)).items():
                fold_scores[k].setdefault(pid, []).append(v)
        if _torch().cuda.is_available():
            _torch().cuda.empty_cache()

    pids = set().union(*[set(fold_scores[a]) for a in OOD_AGGREGATIONS])
    rows = []
    for pid in pids:
        rec = {"patient": pid}
        for a in OOD_AGGREGATIONS:
            vals = [v for v in fold_scores[a].get(pid, []) if np.isfinite(v)]
            rec[f"ood_maha_{a}_{encoder}"] = float(np.mean(vals)) if vals else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    out_dir = BASE / "outputs/ood/ipd_brain"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{encoder}_maha_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out)} patients)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=OOD_ENCODERS)
    ap.add_argument("--bag_size", type=int, default=5000)
    args = ap.parse_args()
    score_encoder(args.encoder, bag_size=args.bag_size)
