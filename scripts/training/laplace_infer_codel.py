"""
Last-layer Laplace approximation uncertainty for codeletion (1p/19q,
IDH-mutant subset) -- codel counterpart of laplace_infer.py. Places a
Gaussian posterior over the already-trained outputs/abmil_codel classifier
head's weights, fit from the curvature of the training loss -- no
retraining, no feature-space density model.

Reuses the cached frozen ABMIL bag embeddings from
extract_codel_embeddings.py (outputs/embeddings_codel/<dataset>/<encoder>/
fold<N>.npz) -- run that first. Only needs the classifier head's weights
(outputs/abmil_codel/<encoder>/fold<N>/best.ckpt) and those cached z
vectors, not a live ABMIL forward pass.

    predictive_entropy (total, from the posterior-predictive mean prob)
        = aleatoric (mean of the per-weight-sample entropies)
        + epistemic_mi (their difference, clipped at 0)

Usage:
    python scripts/training/laplace_infer_codel.py --encoder uni2 --fold 0 --dataset tcga
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from laplace import Laplace

BASE = Path(".")


def load_ebrains_split(encoder: str, fold: int, split_seed: int = 42):
    """Same StratifiedKFold split train_abmil_codel.py used -- fold_info.json
    for the codel deep-ensemble checkpoints already confirms split_seed=42
    reproduces it exactly."""
    d = np.load(BASE / f"outputs/embeddings_codel/ebrains/{encoder}/fold{fold}.npz", allow_pickle=True)
    z, labels = d["z"], d["labels"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
    train_idx, val_idx = list(skf.split(z, labels))[fold]
    return z[train_idx], labels[train_idx], z[val_idx], labels[val_idx]


def binary_entropy(p, eps=1e-10):
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def load_classifier_head(encoder: str, fold: int, latent_dim: int = 256, num_classes: int = 2) -> nn.Linear:
    ckpt = torch.load(BASE / f"outputs/abmil_codel/{encoder}/fold{fold}/best.ckpt",
                       map_location="cpu", weights_only=False)
    head = nn.Linear(latent_dim, num_classes)
    sd = ckpt["model_state"]
    head.weight.data.copy_(sd["classifier.1.weight"])   # classifier = Sequential(Dropout, Linear)
    head.bias.data.copy_(sd["classifier.1.bias"])
    return head, ckpt.get("val_auroc")


def score(la: Laplace, z: torch.Tensor, n_samples: int) -> dict:
    samples = la.predictive_samples(z, n_samples=n_samples).numpy()   # [n_samples, N, 2]
    prob_mutant_samples = samples[:, :, 1]
    mean_prob_mutant = prob_mutant_samples.mean(axis=0)
    predictive_entropy = binary_entropy(mean_prob_mutant)
    per_sample_entropy = binary_entropy(prob_mutant_samples)
    aleatoric = per_sample_entropy.mean(axis=0)
    epistemic_mi = np.clip(predictive_entropy - aleatoric, 0.0, None)
    prob_std = prob_mutant_samples.std(axis=0)
    return {
        "mean_prob_mutant": mean_prob_mutant,
        "predictive_entropy": predictive_entropy,
        "aleatoric": aleatoric,
        "epistemic_mi": epistemic_mi,
        "prob_std": prob_std,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["uni2", "conch", "hoptimus"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--dataset", choices=["ebrains", "tcga", "ipd_brain"], required=True)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="idh-codel-laplace")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    z_train, y_train, z_val, y_val = load_ebrains_split(args.encoder, args.fold)
    z_train_t = torch.tensor(z_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    z_val_t   = torch.tensor(z_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    head, ckpt_val_auroc = load_classifier_head(args.encoder, args.fold)
    head.eval()

    run_name = args.run_name or f"codel_{args.encoder}_{args.dataset}_fold{args.fold}_laplace"
    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
        name=run_name,
        config={**vars(args), "ckpt_val_auroc": ckpt_val_auroc},
        tags=[args.encoder, args.dataset, f"fold{args.fold}", "codel", "laplace"],
    )

    train_loader = DataLoader(TensorDataset(z_train_t, y_train_t), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(z_val_t,   y_val_t),   batch_size=64, shuffle=False)

    la = Laplace(head, likelihood="classification", subset_of_weights="all", hessian_structure="full")
    la.fit(train_loader)
    la.optimize_prior_precision(method="marglik", val_loader=val_loader)
    print(f"fitted prior precision: {la.prior_precision.item():.4f}")

    d_score = np.load(BASE / f"outputs/embeddings_codel/{args.dataset}/{args.encoder}/fold{args.fold}.npz", allow_pickle=True)
    z_score = torch.tensor(d_score["z"], dtype=torch.float32)
    labels = d_score["labels"]
    patients = d_score["patients"]

    results = score(la, z_score, args.n_samples)
    results["labels"] = labels
    results["patients"] = patients

    auroc = roc_auc_score(labels, results["mean_prob_mutant"])
    print(f"AUROC ({args.dataset}, {args.n_samples} weight-posterior samples): {auroc:.4f}")
    print(f"Mean predictive entropy: {results['predictive_entropy'].mean():.4f}")
    print(f"Mean aleatoric:          {results['aleatoric'].mean():.4f}")
    print(f"Mean epistemic MI:       {results['epistemic_mi'].mean():.4f}")

    wandb.log({
        "auroc": auroc,
        "prior_precision": la.prior_precision.item(),
        "mean_predictive_entropy": results["predictive_entropy"].mean(),
        "mean_aleatoric": results["aleatoric"].mean(),
        "mean_epistemic_mi": results["epistemic_mi"].mean(),
        "mean_prob_std": results["prob_std"].mean(),
    })
    wandb.run.summary["auroc"] = auroc

    out_dir = BASE / f"outputs/laplace_codel/{args.dataset}/{args.encoder}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fold{args.fold}.npz"
    np.savez(out_path, n_samples=args.n_samples, prior_precision=la.prior_precision.item(),
             **results)
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
    artifact = wandb.Artifact(f"laplace-codel-{args.encoder}-{args.dataset}-fold{args.fold}", type="laplace_scores")
    artifact.add_file(str(out_path))
    artifact.add_file(str(csv_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
