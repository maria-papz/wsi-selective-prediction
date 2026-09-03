"""
ABMIL training loop for 1p/19q codeletion classification (IDH-mutant subset
only) on EBRAINS. Mirrors train_abmil.py's IDH pipeline exactly, just with
the codeletion labels from build_1p19q_labels.py as target.

Usage:
    python scripts/training/train_abmil_codel.py --encoder uni2 --fold 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.models.abmil import ABMIL
from src.data.dataset import SlideBagDataset, collate_single
from src.utils.constants import ENCODER_FEATURE_DIM, ENCODER_DIRS
from train_abmil import run_epoch

BASE   = Path(".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder",      choices=["uni2", "conch", "hoptimus"], default="uni2")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--latent_dim",   type=int,   default=256)
    parser.add_argument("--dropout",      type=float, default=0.25)
    parser.add_argument("--accum_steps",  type=int,   default=8)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--bag_size",     type=int,   default=None)
    parser.add_argument("--fold",         type=int,   default=0)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--feature_root", type=str,   default="data/processed/features/ebrains",
                                          help="Root dir containing <encoder_root>/<encoder_subdir> "
                                               "feature subtrees (local staged copy or NFS path).")
    parser.add_argument("--wandb_project",type=str,   default="idh-codel-abmil")
    parser.add_argument("--wandb_entity", type=str,   default=None)
    parser.add_argument("--wandb_mode",   choices=["online","offline","disabled"],
                                          default="online")
    parser.add_argument("--run_name",     type=str,   default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_name = args.run_name or f"codel_{args.encoder}_fold{args.fold}_seed{args.seed}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=run_name,
        config=vars(args),
        tags=[args.encoder, f"fold{args.fold}", "codel"],
    )

    encoder_root, encoder_subdir = ENCODER_DIRS[args.encoder]
    in_dim      = ENCODER_FEATURE_DIM[args.encoder]
    feature_dir = BASE / args.feature_root / encoder_root / encoder_subdir

    # load slide table + codeletion labels (IDH-mutant subset only)
    slide_table = pd.read_csv(BASE / f"data/processed/slide_table_ebrains_{args.encoder}.csv")
    clinical    = pd.read_csv(BASE / "data/raw/ebrains/clinical/ebrains_1p19q_labels.csv")
    slide_table = slide_table.merge(
        clinical[["uuid", "codel_binary"]],
        left_on="PATIENT", right_on="uuid", how="inner"
    ).rename(columns={"codel_binary": "codel_status"})
    slide_table["codel_status"] = slide_table["codel_status"].astype(int)

    print(f"Total slides with codeletion labels: {len(slide_table)}")
    print(slide_table["codel_status"].value_counts())

    # stratified 5-fold split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    splits = list(skf.split(slide_table, slide_table["codel_status"]))
    train_idx, val_idx = splits[args.fold]

    train_table = slide_table.iloc[train_idx].reset_index(drop=True)
    val_table   = slide_table.iloc[val_idx].reset_index(drop=True)
    print(f"Train: {len(train_table)}  Val: {len(val_table)}")

    wandb.config.update({
        "n_train": len(train_table),
        "n_val":   len(val_table),
        "train_class_balance": train_table["codel_status"].value_counts().to_dict(),
        "val_class_balance":   val_table["codel_status"].value_counts().to_dict(),
    })

    train_ds = SlideBagDataset(train_table, feature_dir,
                                label_col="codel_status", bag_size=args.bag_size)
    val_ds   = SlideBagDataset(val_table,   feature_dir,
                                label_col="codel_status", bag_size=None)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                               collate_fn=collate_single, num_workers=4,
                               pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                               collate_fn=collate_single, num_workers=4,
                               pin_memory=True)

    model = ABMIL(in_dim=in_dim, latent_dim=args.latent_dim,
                   num_classes=2, dropout=args.dropout).to(DEVICE)

    # class-weighted loss
    class_counts = train_table["codel_status"].value_counts()
    weights = torch.tensor(
        [1.0 / class_counts.get(c, 1) for c in [0, 1]], dtype=torch.float32
    ).to(DEVICE)
    weights = weights / weights.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    ckpt_dir = BASE / f"outputs/abmil_codel/{args.encoder}/fold{args.fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_auroc      = -1.0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        train_loss, train_auroc = run_epoch(
            model, train_loader, optimizer, criterion,
            train=True, accum_steps=args.accum_steps, scaler=scaler)
        val_loss, val_auroc = run_epoch(
            model, val_loader, optimizer, criterion, train=False)
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_auroc={train_auroc:.4f} | "
              f"val_loss={val_loss:.4f} val_auroc={val_auroc:.4f}")

        wandb.log({
            "epoch":        epoch + 1,
            "train/loss":   train_loss,
            "train/auroc":  train_auroc,
            "val/loss":     val_loss,
            "val/auroc":    val_auroc,
            "lr":           scheduler.get_last_lr()[0],
        })

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch":       epoch,
                "val_auroc":   val_auroc,
                "args":        vars(args),
            }, ckpt_dir / "best.ckpt")
            wandb.run.summary["best_val_auroc"] = best_auroc
            wandb.run.summary["best_epoch"]     = epoch + 1
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(best val_auroc={best_auroc:.4f})")
                break

    print(f"\nBest val_auroc: {best_auroc:.4f}")
    print(f"Checkpoint: {ckpt_dir / 'best.ckpt'}")

    with open(ckpt_dir / "fold_info.json", "w") as f:
        json.dump({
            "train_patients": train_table["PATIENT"].tolist(),
            "val_patients":   val_table["PATIENT"].tolist(),
            "best_val_auroc": best_auroc,
        }, f, indent=2)

    artifact = wandb.Artifact(f"abmil-codel-{args.encoder}-fold{args.fold}", type="model")
    artifact.add_file(str(ckpt_dir / "best.ckpt"))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
