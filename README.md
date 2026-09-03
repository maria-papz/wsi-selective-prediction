# Uncertainty-Aware Selective Prediction for Molecular Biomarkers in Diffuse Glioma

MSc thesis project (UCL) evaluating whether uncertainty-aware computational pathology
improves the reliability of molecular biomarker prediction from routine H&E whole-slide
images under genuine external deployment. Predicts IDH-mutation status and 1p/19q
codeletion from H&E whole-slide images using attention-based multiple instance learning
over pathology foundation model features, then evaluates six uncertainty-estimation
families, importance-weighted conformal prediction, and demographic equity, all under
real external distribution shift rather than held-out splits of the same cohort.

- **Training cohort:** EBRAINS (Digital Brain Tumour Atlas), n=642
- **External evaluation cohorts:** TCGA (n=687) and IPD Brain (n=320, cross-continental)
- **Encoders:** UNI2-h, CONCH, H-optimus-1 (frozen, pretrained)
- **Predictors:** UNI2, CONCH, H-optimus, UNI2+H-optimus, and a 3-encoder ensemble
- **Uncertainty families:** predictive entropy, deep ensembles, MC-dropout, last-layer
  Laplace, epistemic (mutual information), and density-based out-of-distribution (OOD)
  detection

Full methodology and results are in the accompanying thesis (not included in this
repository — see [Data and results availability](#data-and-results-availability)).

## Repository structure

```
src/            Model code: ABMIL, MAF-based OOD density model, Laplace helpers,
                dataset loading, shared constants
scripts/        Analysis notebooks (selective prediction, conformal transport,
                fairness, OOD characterisation, 1p/19q codeletion generalisation)
                scripts/training/   model training and inference entry points
                scripts/eval/       statistical testing utilities
build_figures/  Scripts that render thesis figures from cached outputs
data_prep/      Cohort curation, label construction, and data-download scripts
configs/        STAMP tiling/feature-extraction configs, one per cohort x encoder
manifests/      TCGA/EBRAINS download manifests
requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+ (developed and tested across 3.9 and 3.11).

Feature extraction uses [STAMP](https://github.com/KatherLab/STAMP) (El Nahhas et al.,
*Nature Protocols* 2024) for tiling and preprocessing; it is not vendored here and
should be installed separately per its own instructions. Foundation model checkpoints
(UNI2-h, CONCH, H-optimus-1) are pulled from HuggingFace on first use and cached under
`$HF_HOME`; UNI2-h and CONCH are gated models requiring HuggingFace access approval
from their respective repositories before download will succeed.

## Reproducing the pipeline

Runs roughly in this order; each stage's outputs are cached to `outputs/` and consumed
by the next.

1. **Data acquisition and curation** (`data_prep/`, `manifests/`) — cohort download,
   clinical/demographic curation, IDH and 1p/19q label construction, label-confidence
   tiering for IPD Brain's IHC-based labels.
2. **Preprocessing** — STAMP tiling and feature extraction, configured per cohort x
   encoder via `configs/stamp_*.yaml`.
3. **Training** (`scripts/training/`) — ABMIL classifiers (5-fold CV on EBRAINS), the
   deep-ensemble and class-conditional density (OOD) models, and per-fold inference for
   every uncertainty family.
4. **Analysis** (`scripts/*.ipynb`) — baseline performance, selective prediction,
   conformal prediction and its importance-weighted transport variant, demographic
   equity, and the 1p/19q codeletion replication, each with TCGA-locked and
   IPD-Brain-transported/recalibrated variants.
5. **Figures and tables** (`build_figures/`) — regenerates figures from the cached analysis outputs.

## Data and results availability

Raw data (`data/`), intermediate/model outputs (`outputs/`), and environment/cache
directories are not included in this repository (multi-terabyte, and largely
regenerable by rerunning the pipeline above). Cohort access:

- **TCGA** — open access via the [Genomic Data Commons](https://gdc.cancer.gov);
  no application required for the clinical/imaging data used here.
- **EBRAINS (Digital Brain Tumour Atlas)** — requires registration and acceptance
  of the [EBRAINS Data Use Agreement](https://ebrains.eu/terms-policies) via the
  [EBRAINS Data Proxy](https://ebrains.eu/data/) (dataset ID
  `8fc108ab-e2b4-406f-8999-60269dc1f994`).
- **IPD Brain** — obtained through [India Data](https://india-data.org)
  (Chauhan et al., *Scientific Data* 2024). Access requires registration on that
  platform.

## License

MIT — see [LICENSE](LICENSE). Note this covers the code only; the data-availability
terms above (particularly EBRAINS' and IPD Brain's) apply independently and are not
overridden by this license.
