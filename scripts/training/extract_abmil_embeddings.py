"""
Extract frozen ABMIL slide-level embeddings (z vectors) for NatPN training.

Loads each fold's best.ckpt, runs inference on ALL slides,
saves z vectors + labels + predictions to outputs/embeddings/<dataset>/<encoder>/fold<N>.npz

Usage:
    python scripts/training/extract_embeddings.py --encoder uni2 --dataset ebrains
    python scripts/training/extract_embeddings.py --encoder uni2 --dataset tcga
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.abmil import ABMIL
from src.data.dataset import SlideBagDataset, collate_single
from src.utils.constants import ENCODER_FEATURE_DIM, ENCODER_DIRS

BASE   = Path(".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_slide_table(encoder: str, dataset: str) -> pd.DataFrame:
    slide_table = pd.read_csv(
        BASE / f"data/processed/slide_table_{dataset}_{encoder}.csv"
    )
    if dataset == "ebrains":
        clinical = pd.read_csv(
            BASE / "data/raw/ebrains/clinical/ebrains_idh_cases.csv"
        )
        slide_table = slide_table.merge(
            clinical[["uuid", "idh_binary"]],
            left_on="PATIENT", right_on="uuid", how="left"
        ).rename(columns={"idh_binary": "idh_status"})
    elif dataset == "ipd_brain":
        clinical = pd.read_csv(
            BASE / "data/raw/ipd_brain/clinical/ipd_brain_idh_final.csv"
        )
        slide_table = slide_table.merge(
            clinical[["case_id", "idh_status"]],
            left_on="PATIENT", right_on="case_id", how="left"
        )
    else:
        clinical = pd.read_csv(
            BASE / "data/raw/tcga/clinical/tcga_clinical.csv"
        )
        slide_table = slide_table.merge(
            clinical[["case_id", "idh_status"]],
            left_on="PATIENT", right_on="case_id", how="left"
        )
        slide_table["idh_status"] = slide_table["idh_status"].map(
            {"wildtype": 0, "mutant": 1}
        )

    slide_table = slide_table.dropna(subset=["idh_status"]).reset_index(drop=True)
    slide_table["idh_status"] = slide_table["idh_status"].astype(int)
    return slide_table


def extract(model: ABMIL, slide_table: pd.DataFrame,
            feature_dir: Path) -> dict:
    dataset = SlideBagDataset(slide_table, feature_dir, label_col="idh_status")
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         collate_fn=collate_single, num_workers=4)
    model.eval()

    all_z, all_probs, all_labels, all_patients = [], [], [], []

    with torch.no_grad():
        for feats, label, patient_id in loader:
            feats  = feats.to(DEVICE)
            logits, z = model(feats)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()

            all_z.append(z.cpu().numpy())
            all_probs.append(probs)
            all_labels.append(label.item())
            all_patients.append(patient_id)

    return {
        "z":        np.stack(all_z),            # [N, latent_dim]
        "probs":    np.stack(all_probs),         # [N, num_classes]
        "labels":   np.array(all_labels),        # [N]
        "patients": np.array(all_patients),      # [N]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["uni2", "conch", "hoptimus"], required=True)
    parser.add_argument("--dataset", choices=["ebrains", "tcga", "ipd_brain"],           required=True)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--dropout",    type=float, default=0.25)
    parser.add_argument("--n_folds",    type=int, default=5)
    args = parser.parse_args()

    encoder_root, encoder_subdir = ENCODER_DIRS[args.encoder]
    in_dim      = ENCODER_FEATURE_DIM[args.encoder]
    feature_dir = (BASE / "data/processed/features" / args.dataset
                   / encoder_root / encoder_subdir)

    slide_table = load_slide_table(args.encoder, args.dataset)
    print(f"Slides with labels: {len(slide_table)}")
    print(slide_table["idh_status"].value_counts())

    out_dir = BASE / f"outputs/embeddings/{args.dataset}/{args.encoder}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(args.n_folds):
        ckpt_path = BASE / f"outputs/abmil/{args.encoder}/fold{fold}/best.ckpt"
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}  skipping")
            continue

        print(f"\nFold {fold}  loading {ckpt_path}")
        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model = ABMIL(in_dim=in_dim, latent_dim=args.latent_dim,
                      num_classes=2, dropout=args.dropout).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])

        results  = extract(model, slide_table, feature_dir)
        out_path = out_dir / f"fold{fold}.npz"
        np.savez(out_path, **results)

        print(f"  z shape: {results['z'].shape}")
        print(f"  label distribution: {np.bincount(results['labels'])}")
        print(f"  saved  {out_path}")

    print(f"\nDone. All embeddings in {out_dir}/")


if __name__ == "__main__":
    main()