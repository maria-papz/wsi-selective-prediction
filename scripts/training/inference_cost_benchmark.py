"""
Measures real wall-clock inference cost per uncertainty family, on the same
locked ABMIL checkpoint, for the "inference cost per method" comparison in

Only one (encoder, fold) is benchmarked: the relative multiplier across
families is the portable claim (architecture-driven, not data-driven), not
the absolute millisecond numbers, which are device- and load-dependent.

Families measured:
    1. predictive / cross-fold-cross-encoder -- 1 deterministic forward pass
       (the base cost every family pays once to get prob_mutant; the OOD
       and would-be-evidential families reuse this same pass rather than
       paying it again)
    2. density-based OOD -- 1 extra patch-level flow forward pass, on top of (1)
    3. MC-dropout -- N_MC_PASSES dropout-active forward passes through the
       FULL network (dropout touches internal layers, so every pass redoes
       the whole attention-MIL computation)
    4. deep ensemble -- N_DEEP_ENSEMBLE_MEMBERS independently-trained models,
       1 full pass each
    5. last-layer Laplace -- N_LAPLACE_SAMPLES posterior samples through the
       linear classifier head ONLY, reusing the SAME cached z from (1); no
       extra attention-MIL forward pass at all
    6. evidential/NatPN -- not measured (never finished/integrated, see
       BENCH_METHODS.md 3(e)); architecturally would have been the same
       order as (2)/(5) since it also runs on the frozen z, but it was
       dropped for an accuracy failure, not a cost reason

Usage:
    python scripts/training/inference_cost_benchmark.py
    python scripts/training/inference_cost_benchmark.py --encoder uni2 --fold 0 --n_slides 15
"""
import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from laplace import Laplace
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from src.models.abmil import ABMIL
from src.models.maf import MaskedAutoregressiveFlow
from src.models.natpn import ClassConditionalPatchDensityModel, OrthogonalProjection
from src.models.radial_flow import RadialFlow
from src.utils.constants import ENCODER_DIRS, ENCODER_FEATURE_DIM

BASE = Path(".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ("gbm", "astrocytoma", "oligodendroglioma")


def sync():
    """CUDA kernels launch asynchronously -- time.perf_counter() around an
    un-synchronized GPU op measures dispatch time, not compute time, and the
    first CUDA call on top of that eats a one-time context-init/kernel-compile
    cost that isn't representative of any later call. Both together previously
    made 8 full deep-ensemble passes read as 0.6ms (cheaper than the single
    base pass they're additional to, which is impossible) and MC-dropout read
    as cheaper than a single pass. Call this immediately before starting a
    timer and immediately after the timed op, every time, on GPU."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def pick_bench_slides(encoder: str, n_slides: int) -> list[tuple[str, int]]:
    """Real slides spanning a range of bag sizes, not n_slides random/identical ones."""
    encoder_root, encoder_subdir = ENCODER_DIRS[encoder]
    feature_dir = BASE / "data/processed/features/tcga" / encoder_root / encoder_subdir
    slide_table = pd.read_csv(BASE / f"data/processed/slide_table_tcga_{encoder}.csv")
    sizes = []
    for fn in slide_table["FILENAME"]:
        p = feature_dir / fn
        if p.exists():
            with h5py.File(p, "r") as f:
                sizes.append((fn, f["feats"].shape[0]))
    sizes.sort(key=lambda t: t[1])
    picks_idx = np.linspace(0, len(sizes) - 1, n_slides).astype(int)
    return [sizes[i] for i in picks_idx], feature_dir


def load_ebrains_split(encoder: str, fold: int, split_seed: int = 42):
    d = np.load(BASE / f"outputs/embeddings/ebrains/{encoder}/fold{fold}.npz", allow_pickle=True)
    z, labels = d["z"], d["labels"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
    train_idx, val_idx = list(skf.split(z, labels))[fold]
    return z[train_idx], labels[train_idx], z[val_idx], labels[val_idx]


def find_density_checkpoint(encoder: str, fold: int) -> Path:
    root = BASE / "outputs/class_conditional_density" / encoder / f"fold{fold}"
    return sorted(root.glob("*/best.ckpt"))[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["uni2", "conch", "hoptimus"], default="uni2")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_slides", type=int, default=15)
    parser.add_argument("--n_mc_passes", type=int, default=500)
    parser.add_argument("--n_laplace_samples", type=int, default=200)
    parser.add_argument("--deep_ensemble_inits", type=str, default="43,44,45,46")
    args = parser.parse_args()

    in_dim = ENCODER_FEATURE_DIM[args.encoder]
    inits = [int(s) for s in args.deep_ensemble_inits.split(",")]

    print(f"device: {DEVICE}")
    bench_slides, feature_dir = pick_bench_slides(args.encoder, args.n_slides)
    bag_sizes = [n for _, n in bench_slides]
    print(f"Benchmark slides: n={len(bench_slides)}, bag sizes {min(bag_sizes)}-{max(bag_sizes)} "
          f"(mean {np.mean(bag_sizes):.0f})")

    feats_list = []
    for fn, _ in bench_slides:
        with h5py.File(feature_dir / fn, "r") as f:
            feats_list.append(torch.from_numpy(f["feats"][:]).float().to(DEVICE))

    # 1. single deterministic forward pass ------------------------------------
    ckpt = torch.load(BASE / f"outputs/abmil/{args.encoder}/fold{args.fold}/best.ckpt",
                       map_location=DEVICE, weights_only=True)
    model = ABMIL(in_dim=in_dim, latent_dim=256, num_classes=2, dropout=0.25).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Warmup: the first CUDA call on a given model/op pays a one-time
    # context-init/kernel-compile cost that would otherwise land entirely on
    # whichever slide happens to be timed first, corrupting the mean.
    with torch.no_grad():
        model(feats_list[0])
    sync()

    single_pass_times, cached_z = [], []
    with torch.no_grad():
        for feats in feats_list:
            sync()
            t0 = time.perf_counter()
            _, z = model(feats)
            sync()
            single_pass_times.append(time.perf_counter() - t0)
            cached_z.append(z)
    single_pass_times = np.array(single_pass_times)
    print(f"1. Single deterministic pass: {single_pass_times.mean()*1000:.2f} ms/slide")

    # 2. MC-dropout -------------------------------------------------------------
    model.train()
    mc_times = []
    with torch.no_grad():
        for feats in feats_list:
            sync()
            t0 = time.perf_counter()
            for _ in range(args.n_mc_passes):
                model(feats)
            sync()
            mc_times.append(time.perf_counter() - t0)
    model.eval()
    mc_times = np.array(mc_times)
    print(f"2. MC-dropout ({args.n_mc_passes} passes), own extra compute: {mc_times.mean():.3f} s/slide "
          f"= {mc_times.mean()/single_pass_times.mean():.1f}x single pass "
          f"(total incl. mandatory base pass: {(mc_times.mean()*1000 + single_pass_times.mean()*1000)/1000:.3f} s/slide "
          f"= {(mc_times.mean()*1000 + single_pass_times.mean()*1000)/(single_pass_times.mean()*1000):.1f}x)")

    # 3. deep ensemble ------------------------------------------------------------
    de_members = []
    for init_seed in inits:
        de_ckpt = torch.load(
            BASE / f"outputs/abmil_deep_ensemble/{args.encoder}/fold{args.fold}/init{init_seed}/best.ckpt",
            map_location=DEVICE, weights_only=True,
        )
        m = ABMIL(in_dim=in_dim, latent_dim=256, num_classes=2, dropout=0.25).to(DEVICE)
        m.load_state_dict(de_ckpt["model_state"])
        m.eval()
        de_members.append(m)

    with torch.no_grad():
        for m in de_members:
            m(feats_list[0])
    sync()

    de_times = []
    with torch.no_grad():
        for feats in feats_list:
            sync()
            t0 = time.perf_counter()
            for m in de_members:
                m(feats)
            sync()
            de_times.append(time.perf_counter() - t0)
    de_times = np.array(de_times)
    print(f"3. Deep ensemble ({len(inits)} members), own extra compute: {de_times.mean()*1000:.2f} ms/slide "
          f"= {de_times.mean()/single_pass_times.mean():.1f}x single pass "
          f"(total incl. mandatory base pass: {de_times.mean()*1000 + single_pass_times.mean()*1000:.2f} ms/slide "
          f"= {(de_times.mean()*1000 + single_pass_times.mean()*1000)/(single_pass_times.mean()*1000):.2f}x)")

    # 4. Laplace: reuse the cached z, only the linear head is sampled -----------
    z_train, y_train, z_val, y_val = load_ebrains_split(args.encoder, args.fold)
    head = torch.nn.Linear(256, 2)
    sd = ckpt["model_state"]
    head.weight.data.copy_(sd["classifier.1.weight"])
    head.bias.data.copy_(sd["classifier.1.bias"])
    head.eval()

    t0 = time.perf_counter()
    train_loader = DataLoader(TensorDataset(torch.tensor(z_train, dtype=torch.float32),
                                             torch.tensor(y_train, dtype=torch.long)),
                               batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(z_val, dtype=torch.float32),
                                           torch.tensor(y_val, dtype=torch.long)),
                             batch_size=64, shuffle=False)
    la = Laplace(head, likelihood="classification", subset_of_weights="all", hessian_structure="full")
    la.fit(train_loader)
    la.optimize_prior_precision(method="marglik", val_loader=val_loader)
    laplace_fit_time = time.perf_counter() - t0
    print(f"   (Laplace one-time posterior fit, amortised across all future patients: "
          f"{laplace_fit_time:.2f} s -- not a per-patient cost)")

    laplace_times = []
    with torch.no_grad():
        for z in cached_z:
            z_batch = z.unsqueeze(0).cpu()
            t0 = time.perf_counter()
            la.predictive_samples(z_batch, n_samples=args.n_laplace_samples)
            laplace_times.append(time.perf_counter() - t0)
    laplace_times = np.array(laplace_times)
    print(f"4. Laplace ({args.n_laplace_samples} posterior samples, on top of the SAME z "
          f"from step 1), own extra compute: {laplace_times.mean()*1000:.2f} ms/slide "
          f"= {laplace_times.mean()/single_pass_times.mean():.2f}x single pass "
          f"(total incl. mandatory base pass -- Laplace's true floor, never below 1.0x: "
          f"{laplace_times.mean()*1000 + single_pass_times.mean()*1000:.2f} ms/slide "
          f"= {(laplace_times.mean()*1000 + single_pass_times.mean()*1000)/(single_pass_times.mean()*1000):.2f}x)")

    # 5. density-based OOD: one flow forward pass over all patches --------------
    d_ckpt = torch.load(find_density_checkpoint(args.encoder, args.fold),
                         map_location=DEVICE, weights_only=False)
    in_dim_d, latent_d = int(d_ckpt["in_dim"]), int(d_ckpt["latent_dim"])
    ftype, flayers = str(d_ckpt["args"]["flow_type"]), int(d_ckpt["args"]["flow_layers"])
    proj = OrthogonalProjection(in_dim=in_dim_d, latent_dim=latent_d, trainable=False)
    if ftype == "maf":
        flows = [MaskedAutoregressiveFlow(dim=latent_d, num_layers=flayers, use_batch_norm=False)
                 for _ in range(len(CLASS_NAMES))]
    else:
        flows = [RadialFlow(dim=latent_d, num_layers=flayers) for _ in range(len(CLASS_NAMES))]
    density_model = ClassConditionalPatchDensityModel(
        encoder=proj, flows=flows, latent_dim=latent_d,
        num_classes=len(CLASS_NAMES), class_names=CLASS_NAMES).to(DEVICE)
    density_model.load_state_dict(d_ckpt["model_state"])
    density_model.eval()

    chunk_size = 4096
    with torch.no_grad():
        density_model(feats_list[0][:chunk_size])
    sync()

    ood_times = []
    with torch.no_grad():
        for feats in feats_list:
            sync()
            t0 = time.perf_counter()
            chunks = [density_model(feats[s:s + chunk_size]).detach()
                      for s in range(0, len(feats), chunk_size)]
            torch.cat(chunks, dim=0)
            sync()
            ood_times.append(time.perf_counter() - t0)
    ood_times = np.array(ood_times)
    print(f"5. Density-based OOD (flow forward over all patches), own extra compute: "
          f"{ood_times.mean()*1000:.2f} ms/slide = {ood_times.mean()/single_pass_times.mean():.2f}x single pass "
          f"(total incl. mandatory base pass: {ood_times.mean()*1000 + single_pass_times.mean()*1000:.2f} ms/slide "
          f"= {(ood_times.mean()*1000 + single_pass_times.mean()*1000)/(single_pass_times.mean()*1000):.2f}x)")

    # ---- assemble + save --------------------------------------------------------
    base_ms = single_pass_times.mean() * 1000

    def total_row(family, per_patient_op, passes_per_slide, own_extra_ms, note):
        total_ms = base_ms + own_extra_ms
        return dict(family=family, per_patient_op=per_patient_op,
                    passes_per_slide=passes_per_slide,
                    own_extra_ms_per_slide=own_extra_ms,
                    total_ms_per_slide=total_ms,
                    relative_to_single_pass=total_ms / base_ms,
                    note=note)

    rows = [
        dict(family="predictive / cross-fold-cross-encoder",
             per_patient_op="1 deterministic forward pass",
             passes_per_slide=1, own_extra_ms_per_slide=0.0,
             total_ms_per_slide=base_ms, relative_to_single_pass=1.0,
             note="the mandatory base pass every family sits on top of -- produces prob_mutant; "
                  "the full ensemble predictor = 15x this (3 encoders x 5 folds), amortisable across a batch"),
        total_row("density-based OOD",
                  "base pass + 1 flow forward pass over all patches",
                  2, ood_times.mean() * 1000,
                  "separate patch-level pass, not bag-level, on top of the mandatory base pass; "
                  "scales with n_patches like the base pass does"),
        total_row("MC-dropout",
                  f"base pass + {args.n_mc_passes} dropout-active forward passes through the full network",
                  1 + args.n_mc_passes, mc_times.mean() * 1000,
                  "most expensive family measured -- every one of its own passes redoes the "
                  "full attention-MIL forward; the base pass barely moves this total"),
        total_row("deep ensemble",
                  f"base pass + {len(inits)} independently-trained models, 1 full pass each",
                  1 + len(inits), de_times.mean() * 1000,
                  "cheaper than MC-dropout at inference despite needing 4x the training "
                  "compute up front to produce those checkpoints"),
        total_row("last-layer Laplace",
                  f"base pass (reused for its cached z) + {args.n_laplace_samples} posterior "
                  "samples through the linear head only",
                  1, laplace_times.mean() * 1000,
                  f"cheapest ADD-ON by far -- its own extra compute skips the attention-MIL "
                  f"forward entirely, reusing the SAME z the base pass already produced, so its "
                  f"total is only marginally above the mandatory base pass; one-time posterior "
                  f"fit ({laplace_fit_time:.2f}s) is a calibration-time cost, not a per-patient one"),
        dict(family="evidential / NatPN (abandoned)",
             per_patient_op="base pass (reused for its cached z) + 1 small density-net forward "
                            "on frozen z (not measured -- never finished)",
             passes_per_slide=1, own_extra_ms_per_slide=np.nan,
             total_ms_per_slide=np.nan, relative_to_single_pass=np.nan,
             note="architecturally would have been the same order as Laplace (cheap add-on, "
                  "reuses the same z); dropped for an accuracy failure (BENCH_METHODS.md 3e), "
                  "not a cost reason"),
    ]
    table = pd.DataFrame(rows)
    out_path = BASE / "outputs/selective_prediction_bench/inference_cost_by_family.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(f"(Measured on {DEVICE}, n={args.n_slides} real TCGA slides, bag sizes "
          f"{min(bag_sizes)}-{max(bag_sizes)} patches, {args.encoder} fold{args.fold} only -- "
          f"the relative multipliers are the portable claim, not the absolute ms numbers.)")


if __name__ == "__main__":
    main()
