"""
External evaluation of EBRAINS-trained codeletion (1p/19q, IDH-mutant
subset) ABMIL models on TCGA, held out entirely from training.

Loads each fold's best.ckpt (trained by train_abmil_codel.py), runs
inference on the full TCGA codeletion-labelled cohort, and reports
per-fold + ensembled AUROC with a bootstrap CI.

Usage:
    python scripts/training/eval_codel_tcga.py --encoder uni2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.models.abmil import ABMIL
from src.data.dataset import SlideBagDataset, collate_single
from src.utils.constants import ENCODER_FEATURE_DIM, ENCODER_DIRS

BASE   = Path(".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tcga_codel(encoder: str) -> pd.DataFrame:
    slide_table = pd.read_csv(BASE / f"data/processed/slide_table_tcga_{encoder}.csv")
    clinical    = pd.read_csv(BASE / "data/raw/tcga/clinical/tcga_1p19q_labels.csv")
    slide_table = slide_table.merge(
        clinical[["PATIENT", "codel_binary"]],
        on="PATIENT", how="inner"
    ).rename(columns={"codel_binary": "codel_status"})
    slide_table["codel_status"] = slide_table["codel_status"].astype(int)
    return slide_table


def infer(model: ABMIL, slide_table: pd.DataFrame, feature_dir: Path) -> dict:
    dataset = SlideBagDataset(slide_table, feature_dir, label_col="codel_status")
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                          collate_fn=collate_single, num_workers=4)
    model.eval()

    probs, labels, patients = [], [], []
    with torch.no_grad():
        for feats, label, patient_id in loader:
            feats = feats.to(DEVICE)
            logits, _z = model(feats)
            p = F.softmax(logits, dim=-1).cpu().numpy()
            probs.append(p[1])
            labels.append(label.item())
            patients.append(patient_id)

    return {
        "probs":    np.array(probs),
        "labels":   np.array(labels),
        "patients": np.array(patients),
    }


def bootstrap_auroc_ci(labels, probs, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(labels[idx], probs[idx]))
    return np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder",    choices=["uni2", "conch", "hoptimus"], default="uni2")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--dropout",    type=float, default=0.25)
    parser.add_argument("--n_folds",    type=int, default=5)
    parser.add_argument("--feature_root", type=str, default="data/processed/features/tcga",
                                          help="Root dir containing <encoder_root>/<encoder_subdir> "
                                               "feature subtrees (local staged copy or NFS path).")
    args = parser.parse_args()

    encoder_root, encoder_subdir = ENCODER_DIRS[args.encoder]
    in_dim      = ENCODER_FEATURE_DIM[args.encoder]
    feature_dir = BASE / args.feature_root / encoder_root / encoder_subdir

    slide_table = load_tcga_codel(args.encoder)
    print(f"TCGA codeletion-labelled slides: {len(slide_table)}")
    print(slide_table["codel_status"].value_counts())

    fold_results = []
    pooled_probs = np.zeros(len(slide_table))

    for fold in range(args.n_folds):
        ckpt_path = BASE / f"outputs/abmil_codel/{args.encoder}/fold{fold}/best.ckpt"
        if not ckpt_path.exists():
            print(f"[fold {fold}] checkpoint not found at {ckpt_path}, skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = ABMIL(in_dim=in_dim, latent_dim=args.latent_dim,
                       num_classes=2, dropout=args.dropout).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])

        out = infer(model, slide_table, feature_dir)
        auroc = roc_auc_score(out["labels"], out["probs"])
        lo, hi = bootstrap_auroc_ci(out["labels"], out["probs"])
        print(f"[fold {fold}] TCGA external AUROC = {auroc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
              f"(train val_auroc was {ckpt['val_auroc']:.4f})")

        fold_results.append({
            "fold": fold,
            "tcga_auroc": auroc,
            "tcga_auroc_ci95": [lo, hi],
            "ebrains_val_auroc": ckpt["val_auroc"],
        })
        pooled_probs += out["probs"]

    if not fold_results:
        print("No checkpoints found — run train_abmil_codel.py first.")
        return

    pooled_probs /= len(fold_results)
    labels = out["labels"]  # same order/labels across folds (same slide_table)
    ensemble_auroc = roc_auc_score(labels, pooled_probs)
    lo, hi = bootstrap_auroc_ci(labels, pooled_probs)

    fold_aurocs = [r["tcga_auroc"] for r in fold_results]
    print("\n=== Summary ===")
    print(f"Per-fold TCGA AUROC: mean={np.mean(fold_aurocs):.4f} std={np.std(fold_aurocs):.4f} "
          f"({[round(a,4) for a in fold_aurocs]})")
    print(f"Ensemble (mean-prob) TCGA AUROC = {ensemble_auroc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    out_dir = BASE / f"outputs/abmil_codel/{args.encoder}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tcga_external_eval.json", "w") as f:
        json.dump({
            "n_tcga": len(slide_table),
            "tcga_class_balance": slide_table["codel_status"].value_counts().to_dict(),
            "fold_results": fold_results,
            "fold_auroc_mean": float(np.mean(fold_aurocs)),
            "fold_auroc_std": float(np.std(fold_aurocs)),
            "ensemble_auroc": float(ensemble_auroc),
            "ensemble_auroc_ci95": [float(lo), float(hi)],
        }, f, indent=2)
    print(f"\nWrote {out_dir / 'tcga_external_eval.json'}")


if __name__ == "__main__":
    main()
