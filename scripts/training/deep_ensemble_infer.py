"""
Deep-ensemble uncertainty for a locked (encoder, fold)'s init-seed members.

Loads all --init_seeds checkpoints already trained by train_abmil_deep_ensemble.py
for one encoder/fold (each a full, independently-initialised and independently-trained
ABMIL, sharing the exact same EBRAINS train/val partition , and runs one deterministic (dropout off) forward pass per member per slide.


Usage:
    python scripts/training/deep_ensemble_infer.py --encoder uni2 --fold 0 --dataset tcga
    python scripts/training/deep_ensemble_infer.py --encoder conch --fold 2 --dataset tcga \
        --init_seeds 43,44,45,46
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score
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


def binary_entropy(p, eps=1e-10):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


@torch.no_grad()
def deterministic_pass(model: ABMIL, feats: torch.Tensor) -> float:
    model.eval()
    logits, _z = model(feats)
    return F.softmax(logits, dim=-1)[1].item()


def load_member(encoder: str, fold: int, init_seed: int, in_dim: int) -> ABMIL:
    ckpt_path = BASE / f"outputs/abmil_deep_ensemble/{encoder}/fold{fold}/init{init_seed}/best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No deep-ensemble checkpoint at {ckpt_path}; "
            "run train_abmil_deep_ensemble.py (or run_deep_ensemble_sweep.sh) first"
        )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model = ABMIL(in_dim=in_dim,
                  latent_dim=int(ckpt_args.get("latent_dim", 256)),
                  num_classes=2,
                  dropout=float(ckpt_args.get("dropout", 0.25))).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt.get("val_auroc"), ckpt_args


def run_deep_ensemble(members: list[ABMIL], slide_table: pd.DataFrame,
                       feature_dir: Path, bag_size: int | None) -> dict:
    dataset = SlideBagDataset(slide_table, feature_dir,
                               label_col="idh_status", bag_size=bag_size)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                          collate_fn=collate_single, num_workers=4)

    all_patients, all_labels, all_member_probs = [], [], []

    for feats, label, patient_id in loader:
        feats = feats.to(DEVICE)
        member_probs = np.array(
            [deterministic_pass(m, feats) for m in members], dtype=np.float64
        )
        all_patients.append(patient_id)
        all_labels.append(label.item())
        all_member_probs.append(member_probs)

    member_probs = np.stack(all_member_probs)          # [N, K]
    mean_prob_mutant   = member_probs.mean(axis=1)      # [N]
    predictive_entropy = binary_entropy(mean_prob_mutant)
    per_member_entropy = binary_entropy(member_probs)   # [N, K]
    aleatoric          = per_member_entropy.mean(axis=1)
    epistemic_mi       = np.clip(predictive_entropy - aleatoric, 0.0, None)
    prob_std           = member_probs.std(axis=1)

    return {
        "patients":           np.array(all_patients),
        "labels":             np.array(all_labels),
        "member_probs":       member_probs,
        "mean_prob_mutant":   mean_prob_mutant,
        "predictive_entropy": predictive_entropy,
        "aleatoric":          aleatoric,
        "epistemic_mi":       epistemic_mi,
        "prob_std":           prob_std,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder",   choices=["uni2", "conch", "hoptimus"], required=True)
    parser.add_argument("--fold",      type=int, required=True)
    parser.add_argument("--dataset",   choices=["ebrains", "tcga", "ipd_brain"], required=True)
    parser.add_argument("--init_seeds", type=str, default="43,44,45,46",
                         help="Comma-separated init_seeds trained by "
                              "train_abmil_deep_ensemble.py for this encoder/fold.")
    parser.add_argument("--bag_size",  type=int, default=None)
    parser.add_argument("--feature_root", type=str, default="data/processed/features",
                         help="Root containing <dataset>/<encoder_root>/<encoder_subdir>/*.h5. "
                              "Override to a local-disk staged copy to avoid NFS contention "
                              "when scoring several folds concurrently.")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="idh-deep-ensemble-infer")
    parser.add_argument("--wandb_entity",  type=str, default=None)
    parser.add_argument("--wandb_mode",    choices=["online", "offline", "disabled"],
                                            default="online")
    parser.add_argument("--run_name",  type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    init_seeds = [int(s) for s in args.init_seeds.split(",")]

    encoder_root, encoder_subdir = ENCODER_DIRS[args.encoder]
    in_dim      = ENCODER_FEATURE_DIM[args.encoder]

    members, val_aurocs, ckpt_args_list = [], [], []
    for init_seed in init_seeds:
        model, val_auroc, ckpt_args = load_member(args.encoder, args.fold, init_seed, in_dim)
        members.append(model)
        val_aurocs.append(val_auroc)
        ckpt_args_list.append(ckpt_args)

    # Sanity check: all members should share the same architecture (only weight
    # init / training-time randomness should differ across init_seed).
    for key in ("latent_dim", "dropout"):
        vals = {a.get(key) for a in ckpt_args_list}
        if len(vals) > 1:
            raise RuntimeError(
                f"Deep-ensemble members disagree on '{key}': {vals} -- "
                "these should all be the same architecture, only init_seed should differ."
            )

    run_name = args.run_name or f"{args.encoder}_{args.dataset}_fold{args.fold}_de{len(init_seeds)}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=run_name,
        config={**vars(args), "init_seeds": init_seeds, "member_val_aurocs": val_aurocs},
        tags=[args.encoder, args.dataset, f"fold{args.fold}", "deep_ensemble"],
    )

    feature_dir = BASE / args.feature_root / args.dataset / encoder_root / encoder_subdir
    slide_table = load_slide_table(args.encoder, args.dataset)
    print(f"Slides with labels: {len(slide_table)}")
    print(slide_table["idh_status"].value_counts())
    wandb.config.update({"n_patients": len(slide_table)})

    results = run_deep_ensemble(members, slide_table, feature_dir, bag_size=args.bag_size)

    labels = results["labels"]
    auroc_mean = roc_auc_score(labels, results["mean_prob_mutant"])
    print(f"AUROC deep-ensemble mean ({len(init_seeds)} members): {auroc_mean:.4f}")
    print(f"Mean predictive entropy: {results['predictive_entropy'].mean():.4f}")
    print(f"Mean aleatoric:          {results['aleatoric'].mean():.4f}")
    print(f"Mean epistemic MI:       {results['epistemic_mi'].mean():.4f}")

    wandb.log({
        "auroc_mean":              auroc_mean,
        "mean_predictive_entropy": results["predictive_entropy"].mean(),
        "mean_aleatoric":          results["aleatoric"].mean(),
        "mean_epistemic_mi":       results["epistemic_mi"].mean(),
        "mean_prob_std":           results["prob_std"].mean(),
    })
    wandb.run.summary["auroc_mean"] = auroc_mean

    out_dir = BASE / f"outputs/deep_ensemble/{args.dataset}/{args.encoder}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fold{args.fold}.npz"
    np.savez(out_path, init_seeds=np.array(init_seeds), **{k: v for k, v in results.items()})
    print(f"Saved {out_path}")

    summary_df = pd.DataFrame({
        "patient":            results["patients"],
        "label":              results["labels"],
        "mean_prob_mutant":   results["mean_prob_mutant"],
        "predictive_entropy": results["predictive_entropy"],
        "aleatoric":          results["aleatoric"],
        "epistemic_mi":       results["epistemic_mi"],
        "prob_std":           results["prob_std"],
    })
    csv_path = out_dir / f"fold{args.fold}_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    wandb.log({"per_patient_summary": wandb.Table(dataframe=summary_df)})

    artifact = wandb.Artifact(
        f"deep-ensemble-{args.encoder}-{args.dataset}-fold{args.fold}", type="deep_ensemble_scores"
    )
    artifact.add_file(str(out_path))
    artifact.add_file(str(csv_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
