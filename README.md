# PG-SANN: Physics-Guided Self-Attention Neural Network for Perovskite Band-Gap Prediction

A physics-informed tabular Transformer (FT-Transformer style) that predicts
ABX₃ / A₂BB′X₆ perovskite band gaps. PG-SANN augments a standard
Feature-Tokenizer Transformer with a composite physics loss that enforces three
solid-state constraints — **non-negative gaps**, **tolerance-factor
(crystallographic) stability**, and **B-site electronegativity monotonicity** —
so the model stays physically valid *without* post-hoc prediction clamping.

- **Raw (unclamped) negative predictions:** ≈0.5–5.7% vs 12–25% for a
  matched-architecture standard transformer on zero-gap-skewed datasets.
- **Electronegativity monotonicity** (∂E_g/∂χ_B ≤ 0) respected on the large
  majority of samples, unlike the unconstrained baseline.
- **Accuracy parity or better** vs the standard transformer and competitive
  with boosting baselines.

## Highlights

| | Standard Transformer | **PG-SANN** |
|---|---|---|
| Raw NP% (Dataset A) | 24.8 ± 7.3 | **0.5 ± 0.2** |
| Raw NP% (Dataset B) | 12.1 ± 3.1 | **5.7 ± 5.1** |
| Raw NP% (Dataset C) | 20.5 ± 6.7 | **3.6 ± 1.2** |
| Monotonicity violations | 36–57% | **21–32%** |
| R² (parity / better) | baseline | ≈ or > |

## Installation

Python 3.12 (Apple Silicon MPS supported; CPU also works).

```bash
conda create -n pgsann python=3.12 -y
conda activate pgsann
pip install -r requirements.txt
```

## Repository layout

```
src/data/        Loaders, feature engineering (tolerance t, octahedral μ), splits
src/baselines/   RF, SVR, XGBoost, LightGBM, CatBoost, MLP benchmark harness
src/models/      PG-SANN architecture, physics losses, dataset, trainer, Optuna tuning
src/analysis/    Figures, physics-consistency proof, attention, SHAP
configs/         YAML configs (default + Optuna-tuned per dataset)
tests/           pytest suite
```

## Quick start

```bash
# 1. Prepare a dataset (load, add Magpie + t/μ, scale, stratified splits)
python3 src/data/prepare.py --dataset dataset_b
python3 src/data/prepare.py --dataset dataset_c
python3 src/data/prepare.py --dataset dataset_d

# 2. Baselines
python3 src/baselines/train_baselines.py --dataset dataset_b

# 3. Train PG-SANN (physics losses on by default) and the ablation ST
python3 src/models/train_pg_sann.py --config configs/tuned/dataset_b_physics.yaml --dataset dataset_b --tag pg_seed1
python3 src/models/train_pg_sann.py --config configs/tuned/dataset_b_physics.yaml --dataset dataset_b --standard-transformer --tag st_seed1

# 4. Hyperparameter tuning (Optuna)
python3 src/models/tune.py --dataset dataset_b --trials 50

# 5. Physics-consistency proof metrics (raw, unclamped)
python3 src/analysis/physics_proof.py --dataset dataset_b \
    --phys-tags pg_seed1 --std-tags st_seed1

# 6. Run the tests
python3 -m pytest tests/ -q
```

## Physics loss

The composite loss is

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda_1 \mathcal{L}_{\text{bounds}}
+ \lambda_2 \mathcal{L}_{\text{tolerance}} + \lambda_3 \mathcal{L}_{\text{monotonicity}},$$

where:

- **Bounds** — a linear hinge `mean(max(0, -Ê_g))` that actively pushes raw
  predictions onto the non-negative physical boundary (constant gradient for
  Ê_g < 0, zero for Ê_g > 0). This is what makes predictions valid without
  clamping.
- **Tolerance** — penalizes band-gap predictions for crystallographically
  unstable Goldschmidt tolerance factors (t ∉ [0.75, 1.075]).
- **Monotonicity** — penalizes positive ∂Ê_g/∂χ_B, encoding that higher B-site
  electronegativity lowers the conduction band and narrows the gap.

Because unconstrained tuning collapses the physics weights toward zero,
`λ₁ = λ₂ = 1.0` are locked after Optuna (see `configs/tuned/*_physics.yaml`).

## Datasets

The paper's benchmarks (labeled A, B, C in the manuscript) are:
- **A:** OQMD ABX₃ perovskites, 16,323 rows, Magpie descriptors, heavy
  zero-gap skew — `github.com/chenebuah/ML_abx3_dataset`.
- **B:** Chenebuah combined data, 1,453 rows, target `Eg` —
  `github.com/chenebuah/perovskite-ML`.
- **C:** Materials Project perovskites, 9,972 rows — retrieved via the MP API
  (`src/data/download_dataset_d.py`). No `e_above_hull` stability cutoff.

See `data/README.md` for full provenance and the download procedure
(an MP API key is required for Dataset C).

## Tests

32 unit tests cover the physics losses, model, data pipeline, and utilities:

```bash
python3 -m pytest tests/ -q
```

## License

See the repository for license details.
