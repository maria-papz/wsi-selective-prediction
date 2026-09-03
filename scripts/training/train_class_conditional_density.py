from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.models.natpn import (
    ClassConditionalDensityLoss,
    ClassConditionalPatchDensityModel,
    OrthogonalProjection,
)
from src.models.maf import MaskedAutoregressiveFlow
from src.models.radial_flow import RadialFlow


BASE = Path(".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_DIRS = {
    "uni2": ("uni", "uni2-49b04e14"),
    "conch": ("conch", "conch-49b04e14"),
    "hoptimus": ("hoptimus", "h-optimus-1-49b04e14"),
}

ENCODER_FEATURE_DIM = {
    "uni2": 1536,
    "conch": 512,
    "hoptimus": 1536,
}

CLASS_NAMES = (
    "gbm",
    "astrocytoma",
    "oligodendroglioma",
)
NUM_CLASSES = len(CLASS_NAMES)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class PatchDataset(Dataset):
    def __init__(
        self,
        slide_table: pd.DataFrame,
        feature_dir: Path,
        label_col: str,
        bag_size: int | None,
    ):
        self.slide_table = slide_table.reset_index(drop=True)
        self.feature_dir = feature_dir
        self.label_col = label_col
        self.bag_size = bag_size

    def __len__(self) -> int:
        return len(self.slide_table)

    def __getitem__(self, idx: int):
        row = self.slide_table.iloc[idx]
        patient = str(row["PATIENT"])
        h5_path = self.feature_dir / f"{patient}.h5"

        if not h5_path.exists():
            raise FileNotFoundError(f"Missing feature file: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            if "feats" not in f:
                raise KeyError(f"'feats' dataset missing in: {h5_path}")
            feats = f["feats"][:]

        if len(feats) == 0:
            raise RuntimeError(f"No patch features found in: {h5_path}")

        if self.bag_size is not None and len(feats) > self.bag_size:
            selected = torch.randperm(len(feats))[: self.bag_size].numpy()
            feats = feats[selected]

        label = int(row[self.label_col])

        return (
            torch.from_numpy(feats).float(),
            torch.tensor(label, dtype=torch.long),
        )


def collate_patches(batch):
    all_feats = []
    all_labels = []

    for feats, label in batch:
        all_feats.append(feats)
        all_labels.append(label.expand(len(feats)))

    return (
        torch.cat(all_feats, dim=0),
        torch.cat(all_labels, dim=0),
    )


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    persistent_workers: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=seed_worker,
        collate_fn=collate_patches,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(persistent_workers and num_workers > 0),
    )


def make_flow(
    flow_type: str,
    latent_dim: int,
    flow_layers: int,
):
    if flow_type == "maf":
        return MaskedAutoregressiveFlow(
            dim=latent_dim,
            num_layers=flow_layers,
            use_batch_norm=False,
        )

    if flow_type == "radial":
        return RadialFlow(
            dim=latent_dim,
            num_layers=flow_layers,
        )

    raise ValueError(f"Unknown flow type: {flow_type}")


def build_model(
    in_dim: int,
    latent_dim: int,
    flow_type: str,
    flow_layers: int,
) -> ClassConditionalPatchDensityModel:
    # Fixed projection: the latent representation must remain stationary
    # while the flows learn class-conditional densities.
    encoder = OrthogonalProjection(
        in_dim=in_dim,
        latent_dim=latent_dim,
        trainable=False,
    )

    flows = [
        make_flow(
            flow_type=flow_type,
            latent_dim=latent_dim,
            flow_layers=flow_layers,
        )
        for _ in range(NUM_CLASSES)
    ]

    return ClassConditionalPatchDensityModel(
        encoder=encoder,
        flows=flows,
        latent_dim=latent_dim,
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
    )


@torch.no_grad()
def collect_raw_latents(
    model: ClassConditionalPatchDensityModel,
    loader: DataLoader,
    max_patches: int,
) -> torch.Tensor:
    model.eval()

    chunks = []
    collected = 0

    for feats, _ in loader:
        feats = feats.to(DEVICE, non_blocking=True)
        raw_z = model.encode_raw(feats).detach().cpu()

        remaining = max_patches - collected
        if remaining <= 0:
            break

        raw_z = raw_z[:remaining]
        chunks.append(raw_z)
        collected += len(raw_z)

        if collected >= max_patches:
            break

    if not chunks:
        raise RuntimeError(
            "No patches were available for latent normalization."
        )

    return torch.cat(chunks, dim=0)


def run_epoch(
    model: ClassConditionalPatchDensityModel,
    loader: DataLoader,
    criterion: ClassConditionalDensityLoss,
    optimizer: torch.optim.Optimizer | None,
    train: bool,
):
    if train:
        if optimizer is None:
            raise ValueError("optimizer is required when train=True")
        model.train()
    else:
        model.eval()

    total_optimization_loss = 0.0
    total_patches = 0

    selected_log_probs = []
    all_labels = []
    all_density_probs = []

    class_log_prob_sums = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )
    class_counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
    )

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for feats, labels in loader:
            feats = feats.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            class_log_probs = model(feats)
            loss, selected_lp = criterion(
                class_log_probs,
                labels,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss encountered: {loss.item()}"
                )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

            density_probs = (
                model.density_class_probabilities_from_log_probs(
                    class_log_probs
                )
            )

            n_patches = len(feats)

            total_optimization_loss += (
                loss.item() * n_patches
            )
            total_patches += n_patches

            labels_cpu = labels.detach().cpu()
            selected_cpu = selected_lp.detach().cpu()

            selected_log_probs.append(selected_cpu)
            all_labels.append(labels_cpu)
            all_density_probs.append(
                density_probs.detach().cpu()
            )

            for class_idx in range(NUM_CLASSES):
                mask = labels_cpu == class_idx

                if mask.any():
                    class_log_prob_sums[class_idx] += float(
                        selected_cpu[mask].sum().item()
                    )
                    class_counts[class_idx] += int(
                        mask.sum().item()
                    )

    if total_patches == 0:
        raise RuntimeError(
            "The data loader produced zero patches."
        )

    selected_log_probs_np = torch.cat(
        selected_log_probs
    ).numpy()

    labels_np = torch.cat(all_labels).numpy()

    density_probs_np = torch.cat(
        all_density_probs
    ).numpy()

    per_class_nll = np.full(
        NUM_CLASSES,
        np.nan,
        dtype=np.float64,
    )

    for class_idx in range(NUM_CLASSES):
        if class_counts[class_idx] > 0:
            mean_lp = (
                class_log_prob_sums[class_idx]
                / class_counts[class_idx]
            )
            per_class_nll[class_idx] = -mean_lp

    valid_nll = per_class_nll[
        ~np.isnan(per_class_nll)
    ]

    if len(valid_nll) != NUM_CLASSES:
        raise RuntimeError(
            "At least one tumour class produced no patches "
            "during this epoch."
        )

    balanced_nll = float(valid_nll.mean())

    predictions = density_probs_np.argmax(axis=1)

    accuracy = accuracy_score(
        labels_np,
        predictions,
    )

    try:
        auroc = roc_auc_score(
            labels_np,
            density_probs_np,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auroc = float("nan")

    return {
        "optimization_loss": (
            total_optimization_loss / total_patches
        ),
        "density_loss": balanced_nll,
        "selected_log_prob_mean": float(
            selected_log_probs_np.mean()
        ),
        "selected_log_prob_std": float(
            selected_log_probs_np.std()
        ),
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "per_class_nll": per_class_nll,
        "class_counts": class_counts,
        "n_patches": int(total_patches),
    }


def class_counts_from_slide_table(
    slide_table: pd.DataFrame,
) -> torch.Tensor:
    counts = [
        int(
            (
                slide_table["natpn_label"]
                == class_idx
            ).sum()
        )
        for class_idx in range(NUM_CLASSES)
    ]

    return torch.tensor(
        counts,
        dtype=torch.float32,
    )


def format_per_class_nll(
    values: Sequence[float],
) -> str:
    parts = []

    for class_name, value in zip(
        CLASS_NAMES,
        values,
    ):
        if np.isnan(value):
            parts.append(f"{class_name}=nan")
        else:
            parts.append(
                f"{class_name}={value:.3f}"
            )

    return ", ".join(parts)


def validate_split_integrity(
    train_table: pd.DataFrame,
    val_table: pd.DataFrame,
    fold_info: dict,
) -> None:
    train_patients = set(
        train_table["PATIENT"].astype(str)
    )
    val_patients = set(
        val_table["PATIENT"].astype(str)
    )

    overlap = train_patients & val_patients

    if overlap:
        raise RuntimeError(
            "Train/validation overlap detected: "
            f"{sorted(overlap)[:10]}"
        )

    expected_train = set(
        map(str, fold_info["train_patients"])
    )
    expected_val = set(
        map(str, fold_info["val_patients"])
    )

    missing_train = expected_train - train_patients
    missing_val = expected_val - val_patients

    if missing_train:
        print(
            f"Warning: {len(missing_train)} training cases "
            "were missing after the clinical merge."
        )

    if missing_val:
        print(
            f"Warning: {len(missing_val)} validation cases "
            "were missing after the clinical merge."
        )

    for class_idx, class_name in enumerate(
        CLASS_NAMES
    ):
        n_train = int(
            (
                train_table["natpn_label"]
                == class_idx
            ).sum()
        )

        n_val = int(
            (
                val_table["natpn_label"]
                == class_idx
            ).sum()
        )

        if n_train == 0:
            raise RuntimeError(
                f"No training cases for {class_name}."
            )

        if n_val == 0:
            raise RuntimeError(
                f"No validation cases for {class_name}."
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--encoder",
        choices=["uni2", "conch", "hoptimus"],
        required=True,
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--flow_type",
        choices=["maf", "radial"],
        default="maf",
    )
    parser.add_argument(
        "--flow_layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    # Number of slides per training loader batch.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    # Use one full validation slide per batch by default.
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=1,
    )

    # Training uses random patch subsampling.
    parser.add_argument(
        "--bag_size",
        type=int,
        default=2000,
    )

    # 0 means all validation patches.
    parser.add_argument(
        "--val_bag_size",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--normalizer_patches",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="idh-density",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )

    args = parser.parse_args()

    if args.fold < 0:
        raise ValueError("fold must be non-negative.")

    if args.latent_dim <= 0:
        raise ValueError(
            "latent_dim must be positive."
        )

    if args.flow_layers <= 0:
        raise ValueError(
            "flow_layers must be positive."
        )

    if args.epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if args.val_batch_size <= 0:
        raise ValueError(
            "val_batch_size must be positive."
        )

    if args.normalizer_patches <= 0:
        raise ValueError(
            "normalizer_patches must be positive."
        )

    if args.bag_size <= 0:
        args.bag_size = None

    if args.val_bag_size <= 0:
        args.val_bag_size = None

    run_seed = args.seed + args.fold
    seed_everything(run_seed)

    encoder_root, encoder_subdir = (
        ENCODER_DIRS[args.encoder]
    )

    feature_dir = (
        BASE
        / "data/processed/features/ebrains"
        / encoder_root
        / encoder_subdir
    )

    if not feature_dir.exists():
        raise FileNotFoundError(
            f"Feature directory does not exist: {feature_dir}"
        )

    in_dim = ENCODER_FEATURE_DIM[
        args.encoder
    ]

    clinical_path = (
        BASE
        / "data/raw/ebrains/clinical"
        / "ebrains_idh_cases.csv"
    )

    slide_table_path = (
        BASE
        / "data/processed"
        / f"slide_table_ebrains_{args.encoder}.csv"
    )

    fold_info_path = (
        BASE
        / "outputs/abmil"
        / args.encoder
        / f"fold{args.fold}"
        / "fold_info.json"
    )

    for required_path in (
        clinical_path,
        slide_table_path,
        fold_info_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file does not exist: {required_path}"
            )

    clinical = pd.read_csv(clinical_path)

    def get_label(row):
        tumour_type = (
            str(row["tumour_type"])
            .strip()
            .lower()
        )

        if tumour_type == "gbm":
            return 0

        if tumour_type == "astrocytoma":
            return 1

        if tumour_type == "oligodendroglioma":
            return 2

        return -1

    clinical["natpn_label"] = clinical.apply(
        get_label,
        axis=1,
    )

    clinical_labels = (
        clinical[["uuid", "natpn_label"]]
        .drop_duplicates(subset="uuid")
    )

    slide_table = pd.read_csv(
        slide_table_path
    )

    required_columns = {"PATIENT"}

    missing_columns = required_columns - set(
        slide_table.columns
    )

    if missing_columns:
        raise KeyError(
            "Slide table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    slide_table["PATIENT"] = (
        slide_table["PATIENT"].astype(str)
    )

    clinical_labels["uuid"] = (
        clinical_labels["uuid"].astype(str)
    )

    slide_table = slide_table.merge(
        clinical_labels,
        left_on="PATIENT",
        right_on="uuid",
        how="left",
        validate="many_to_one",
    )

    slide_table = slide_table.dropna(
        subset=["natpn_label"]
    ).reset_index(drop=True)

    slide_table["natpn_label"] = (
        slide_table["natpn_label"].astype(int)
    )

    slide_table = slide_table[
        slide_table["natpn_label"].between(
            0,
            NUM_CLASSES - 1,
        )
    ].reset_index(drop=True)

    with open(fold_info_path) as f:
        fold_info = json.load(f)

    train_ids = set(
        map(str, fold_info["train_patients"])
    )

    val_ids = set(
        map(str, fold_info["val_patients"])
    )

    train_table = slide_table[
        slide_table["PATIENT"].isin(train_ids)
    ].reset_index(drop=True)

    val_table = slide_table[
        slide_table["PATIENT"].isin(val_ids)
    ].reset_index(drop=True)

    if train_table.empty:
        raise RuntimeError(
            "Training table is empty."
        )

    if val_table.empty:
        raise RuntimeError(
            "Validation table is empty."
        )

    validate_split_integrity(
        train_table=train_table,
        val_table=val_table,
        fold_info=fold_info,
    )

    config_name = (
        f"{args.flow_type}_"
        f"ldim{args.latent_dim}_"
        f"layers{args.flow_layers}_"
        f"seed{run_seed}"
    )

    ckpt_dir = (
        BASE
        / "outputs/class_conditional_density"
        / args.encoder
        / f"fold{args.fold}"
        / config_name
    )

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=(
            f"class_cond_density_"
            f"{args.encoder}_"
            f"fold{args.fold}_"
            f"{config_name}"
        ),
        config={
            **vars(args),
            "run_seed": run_seed,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(DEVICE),
            "train_patients": (
                fold_info["train_patients"]
            ),
            "val_patients": (
                fold_info["val_patients"]
            ),
        },
        tags=[
            args.encoder,
            f"fold{args.fold}",
            "class-conditional-density",
            args.flow_type,
        ],
    )

    print(f"Device:        {DEVICE}")
    print(f"Run seed:      {run_seed}")
    print(f"Train slides:  {len(train_table)}")
    print(f"Val slides:    {len(val_table)}")
    print("Train slide counts:")

    for class_idx, class_name in enumerate(
        CLASS_NAMES
    ):
        count = int(
            (
                train_table["natpn_label"]
                == class_idx
            ).sum()
        )
        print(f"  {class_name}: {count}")

    print("Validation slide counts:")

    for class_idx, class_name in enumerate(
        CLASS_NAMES
    ):
        count = int(
            (
                val_table["natpn_label"]
                == class_idx
            ).sum()
        )
        print(f"  {class_name}: {count}")

    train_ds = PatchDataset(
        slide_table=train_table,
        feature_dir=feature_dir,
        label_col="natpn_label",
        bag_size=args.bag_size,
    )

    val_ds = PatchDataset(
        slide_table=val_table,
        feature_dir=feature_dir,
        label_col="natpn_label",
        bag_size=args.val_bag_size,
    )

    normalizer_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=run_seed + 1_000,
        persistent_workers=False,
    )

    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=run_seed + 2_000,
        persistent_workers=True,
    )

    val_loader = make_loader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=run_seed + 3_000,
        persistent_workers=True,
    )

    model = build_model(
        in_dim=in_dim,
        latent_dim=args.latent_dim,
        flow_type=args.flow_type,
        flow_layers=args.flow_layers,
    ).to(DEVICE)

    model.set_class_priors(
        class_counts_from_slide_table(
            train_table
        )
    )

    print(
        "Fitting latent mean/std from "
        "training patches..."
    )

    raw_latents = collect_raw_latents(
        model=model,
        loader=normalizer_loader,
        max_patches=args.normalizer_patches,
    )

    model.fit_latent_normalizer(
        raw_latents
    )

    del raw_latents

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    n_trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Model params: {n_params:,}; "
        f"trainable: {n_trainable:,}; "
        f"classes={NUM_CLASSES}; "
        f"flow={args.flow_type}"
    )

    wandb.config.update(
        {
            "n_params": n_params,
            "n_trainable": n_trainable,
            "in_dim": in_dim,
            "class_names": list(CLASS_NAMES),
            "checkpoint_dir": str(ckpt_dir),
        },
        allow_val_change=True,
    )

    criterion = ClassConditionalDensityLoss(
        balance_classes=True
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "The model has no trainable parameters."
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    best_val_density_loss = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            train=True,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            train=False,
        )

        scheduler.step()

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"train_nll="
            f"{train_metrics['density_loss']:.4f} "
            f"train_logp="
            f"{train_metrics['selected_log_prob_mean']:.2f}"
            f"±"
            f"{train_metrics['selected_log_prob_std']:.2f} "
            f"train_acc="
            f"{train_metrics['accuracy']:.4f} "
            f"train_auc="
            f"{train_metrics['auroc']:.4f} "
            f"train_patches="
            f"{train_metrics['n_patches']} | "
            f"val_nll="
            f"{val_metrics['density_loss']:.4f} "
            f"val_logp="
            f"{val_metrics['selected_log_prob_mean']:.2f}"
            f"±"
            f"{val_metrics['selected_log_prob_std']:.2f} "
            f"val_acc="
            f"{val_metrics['accuracy']:.4f} "
            f"val_auc="
            f"{val_metrics['auroc']:.4f} "
            f"val_patches="
            f"{val_metrics['n_patches']}"
        )

        print(
            "  train per-class NLL: "
            + format_per_class_nll(
                train_metrics["per_class_nll"]
            )
        )

        print(
            "  val   per-class NLL: "
            + format_per_class_nll(
                val_metrics["per_class_nll"]
            )
        )

        log_data = {
            "epoch": epoch + 1,
            "lr": scheduler.get_last_lr()[0],
            "train/optimization_loss": (
                train_metrics["optimization_loss"]
            ),
            "train/density_loss": (
                train_metrics["density_loss"]
            ),
            "train/selected_logp_mean": (
                train_metrics[
                    "selected_log_prob_mean"
                ]
            ),
            "train/selected_logp_std": (
                train_metrics[
                    "selected_log_prob_std"
                ]
            ),
            "train/density_accuracy": (
                train_metrics["accuracy"]
            ),
            "train/density_auroc": (
                train_metrics["auroc"]
            ),
            "train/n_patches": (
                train_metrics["n_patches"]
            ),
            "val/optimization_loss": (
                val_metrics["optimization_loss"]
            ),
            "val/density_loss": (
                val_metrics["density_loss"]
            ),
            "val/selected_logp_mean": (
                val_metrics[
                    "selected_log_prob_mean"
                ]
            ),
            "val/selected_logp_std": (
                val_metrics[
                    "selected_log_prob_std"
                ]
            ),
            "val/density_accuracy": (
                val_metrics["accuracy"]
            ),
            "val/density_auroc": (
                val_metrics["auroc"]
            ),
            "val/n_patches": (
                val_metrics["n_patches"]
            ),
        }

        for class_idx, class_name in enumerate(
            CLASS_NAMES
        ):
            log_data[
                f"train/nll_{class_name}"
            ] = train_metrics[
                "per_class_nll"
            ][class_idx]

            log_data[
                f"val/nll_{class_name}"
            ] = val_metrics[
                "per_class_nll"
            ][class_idx]

        wandb.log(log_data)

        current_val_loss = (
            val_metrics["density_loss"]
        )

        if current_val_loss < best_val_density_loss:
            best_val_density_loss = current_val_loss
            best_epoch = epoch
            no_improve = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in (
                    model.state_dict().items()
                )
            }

            wandb.run.summary[
                "best_val_density_loss"
            ] = best_val_density_loss

            wandb.run.summary[
                "best_epoch"
            ] = epoch + 1

        else:
            no_improve += 1

            if no_improve >= args.patience:
                print(
                    f"Early stopping at "
                    f"epoch {epoch + 1}"
                )
                break

    if best_state is None:
        raise RuntimeError(
            "Training ended without a valid checkpoint."
        )

    model.load_state_dict(best_state)
    model.to(DEVICE)

    checkpoint_path = (
        ckpt_dir / "best.ckpt"
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": best_epoch + 1,
            "val_density_loss": (
                best_val_density_loss
            ),
            "args": vars(args),
            "run_seed": run_seed,
            "in_dim": in_dim,
            "latent_dim": args.latent_dim,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "train_patients": (
                fold_info["train_patients"]
            ),
            "val_patients": (
                fold_info["val_patients"]
            ),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(DEVICE),
            "feature_dir": str(feature_dir),
            "config_name": config_name,
            "ood_calibrated": False,
        },
        checkpoint_path,
    )

    print(
        f"Best val density loss: "
        f"{best_val_density_loss:.4f}"
    )

    print(
        f"Best epoch: {best_epoch + 1}"
    )

    print(
        "OOD thresholds were not calibrated "
        "during training."
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    artifact = wandb.Artifact(
        (
            "class-conditional-density-"
            f"{args.encoder}-"
            f"fold{args.fold}-"
            f"{config_name}"
        ),
        type="model",
    )

    artifact.add_file(
        str(checkpoint_path)
    )

    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
