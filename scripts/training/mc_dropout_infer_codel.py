"""
MC-dropout uncertainty for codeletion (1p/19q, IDH-mutant subset) --
codel counterpart of mc_dropout_infer.py. Uses the already-trained
outputs/abmil_codel/<encoder>/fold<N>/best.ckpt directly -- no retraining
needed, this only adds extra stochastic forward passes at inference time.

    predictive_entropy (total, from the T-pass mean prob)
        = aleatoric (mean of the T per-pass entropies)
        + epistemic_mi (their difference, clipped at 0)

Usage:
    python scripts/training/mc_dropout_infer_codel.py --encoder uni2 --fold 0 --dataset tcga
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
from extract_codel_embeddings import load_slide_table

BASE   = Path(".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_entropy(p, eps=1e-10):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


@torch.no_grad()
def mc_dropout_pass(model: ABMIL, feats: torch.Tensor, n_passes: int) -> np.ndarray:
    probs = np.empty(n_passes, dtype=np.float64)
    for t in range(n_passes):
        logits, _z = model(feats)
        probs[t] = F.softmax(logits, dim=-1)[1].item()
    return probs


@torch.no_grad()
def deterministic_pass(model: ABMIL, feats: torch.Tensor) -> float:
    was_training = model.training
    model.eval()
    logits, _z = model(feats)
    prob = F.softmax(logits, dim=-1)[1].item()
    if was_training:
        model.train()
    return prob


def run_mc_dropout(model: ABMIL, slide_table: pd.DataFrame, feature_dir: Path,
                    n_passes: int, bag_size: int | None) -> dict:
    dataset = SlideBagDataset(slide_table, feature_dir,
                               label_col="codel_status", bag_size=bag_size)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                          collate_fn=collate_single, num_workers=4)

    all_patients, all_labels = [], []
    all_mc_probs, all_det_probs = [], []

    model.train()  # dropout active; no batchnorm in ABMIL, so this is safe
    for feats, label, patient_id in loader:
        feats = feats.to(DEVICE)
        det_prob = deterministic_pass(model, feats)
        mc_probs = mc_dropout_pass(model, feats, n_passes)

        all_patients.append(patient_id)
        all_labels.append(label.item())
        all_det_probs.append(det_prob)
        all_mc_probs.append(mc_probs)

    mc_probs = np.stack(all_mc_probs)              # [N, T]
    mean_prob_mutant   = mc_probs.mean(axis=1)      # [N]
    predictive_entropy = binary_entropy(mean_prob_mutant)
    per_pass_entropy   = binary_entropy(mc_probs)   # [N, T]
    aleatoric          = per_pass_entropy.mean(axis=1)
    epistemic_mi       = np.clip(predictive_entropy - aleatoric, 0.0, None)
    prob_std           = mc_probs.std(axis=1)

    return {
        "patients":           np.array(all_patients),
        "labels":             np.array(all_labels),
        "det_prob_mutant":    np.array(all_det_probs),
        "mc_probs":           mc_probs,
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
    parser.add_argument("--n_passes",  type=int, default=50)
    parser.add_argument("--bag_size",  type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--dropout",    type=float, default=0.25)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--feature_root", type=str, default="data/processed/features",
                         help="Root containing <dataset>/<encoder_root>/<encoder_subdir>/*.h5. "
                              "Override to a local-disk staged copy to avoid NFS contention.")
    parser.add_argument("--wandb_project", type=str, default="idh-codel-mc-dropout")
    parser.add_argument("--wandb_entity",  type=str, default=None)
    parser.add_argument("--wandb_mode",    choices=["online", "offline", "disabled"],
                                            default="online")
    parser.add_argument("--run_name",  type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = BASE / f"outputs/abmil_codel/{args.encoder}/fold{args.fold}/best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No trained checkpoint at {ckpt_path}; run train_abmil_codel.py first")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    latent_dim = int(ckpt_args.get("latent_dim", args.latent_dim))
    dropout    = float(ckpt_args.get("dropout", args.dropout))

    run_name = args.run_name or f"codel_{args.encoder}_{args.dataset}_fold{args.fold}_mc{args.n_passes}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=run_name,
        config={**vars(args), "ckpt_latent_dim": latent_dim, "ckpt_dropout": dropout,
                "ckpt_val_auroc": ckpt.get("val_auroc")},
        tags=[args.encoder, args.dataset, f"fold{args.fold}", "codel", "mc_dropout"],
    )

    encoder_root, encoder_subdir = ENCODER_DIRS[args.encoder]
    in_dim      = ENCODER_FEATURE_DIM[args.encoder]
    feature_dir = BASE / args.feature_root / args.dataset / encoder_root / encoder_subdir

    slide_table = load_slide_table(args.encoder, args.dataset)
    print(f"Slides with codel labels: {len(slide_table)}")
    print(slide_table["codel_status"].value_counts())
    wandb.config.update({"n_patients": len(slide_table)})

    model = ABMIL(in_dim=in_dim, latent_dim=latent_dim,
                  num_classes=2, dropout=dropout).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)

    results = run_mc_dropout(model, slide_table, feature_dir,
                              n_passes=args.n_passes, bag_size=args.bag_size)

    labels = results["labels"]
    auroc_det = roc_auc_score(labels, results["det_prob_mutant"])
    auroc_mc  = roc_auc_score(labels, results["mean_prob_mutant"])
    print(f"AUROC deterministic (dropout off): {auroc_det:.4f}")
    print(f"AUROC MC-dropout mean ({args.n_passes} passes): {auroc_mc:.4f}")
    print(f"Mean predictive entropy: {results['predictive_entropy'].mean():.4f}")
    print(f"Mean aleatoric:          {results['aleatoric'].mean():.4f}")
    print(f"Mean epistemic MI:       {results['epistemic_mi'].mean():.4f}")

    wandb.log({
        "auroc_deterministic": auroc_det,
        "auroc_mc_mean":       auroc_mc,
        "mean_predictive_entropy": results["predictive_entropy"].mean(),
        "mean_aleatoric":          results["aleatoric"].mean(),
        "mean_epistemic_mi":       results["epistemic_mi"].mean(),
        "mean_prob_std":           results["prob_std"].mean(),
    })
    wandb.run.summary["auroc_mc_mean"] = auroc_mc
    wandb.run.summary["auroc_deterministic"] = auroc_det

    out_dir = BASE / f"outputs/mc_dropout_codel/{args.dataset}/{args.encoder}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fold{args.fold}.npz"
    np.savez(out_path, n_passes=args.n_passes, **{k: v for k, v in results.items()})
    print(f"Saved {out_path}")

    summary_df = pd.DataFrame({
        "patient":            results["patients"],
        "label":              results["labels"],
        "det_prob_mutant":    results["det_prob_mutant"],
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
        f"mc-dropout-codel-{args.encoder}-{args.dataset}-fold{args.fold}", type="mc_dropout_scores"
    )
    artifact.add_file(str(out_path))
    artifact.add_file(str(csv_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
