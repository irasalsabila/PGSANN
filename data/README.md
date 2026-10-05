# Data

The benchmark datasets and processed artifacts are available from the Zenodo
data release cited below.

**Citation:** Pranida, S. Z., Krito, J. A., & Hapsari, A. W. (2026).
*PG-SANN Perovskite Band-Gap Prediction Benchmark Datasets* (Version 1.0.0)
[Data set]. Zenodo. https://doi.org/10.5281/zenodo.23162501

This file documents provenance, expected files, and the reproducible download
procedure for every dataset.

## Dataset A — OQMD ABX₃ perovskites (internal path: `dataset_b`)

- **Source:** `github.com/chenebuah/ML_abx3_dataset` (arXiv:2312.11335)
  and `github.com/chenebuah/perovskite-ML` (Mater. Today Commun. 27, 102462, 2021)
- **File:** `dataset_a_oqmd.csv` — **16,323 rows**, Magpie statistical descriptors
  + lattice parameters; target `Eg`.
- **Skew:** 11,316 of 16,323 entries are metallic (`Eg=0`) → heavy zero-gap skew,
  the central challenge PG-SANN addresses.
- **Note:** the manuscript's "45,570 compounds" figure refers to the full raw
  OQMD extraction; the public repos distribute the reproducible 16,323-row file.

## Dataset B — Chenebuah combined data (internal path: `dataset_c`)

- **Source:** `github.com/chenebuah/perovskite-ML` (`combine.zip`)
- **File:** `dataset_b_chenebuah.csv` — **1,453 rows**, Magpie descriptors + `gtf`, `of`,
  `rho`, `Ehull`, `Ef`; target `Eg`.

## Dataset C — Materials Project perovskites (internal path: `dataset_d`)

- **Source:** Materials Project API (`next-gen.materialsproject.org`)
- **Download (API key required):**
  ```bash
  export MP_API_KEY=your_key   # from https://next-gen.materialsproject.org/api
  python3 src/data/download_dataset_d.py
  ```
- **File:** `dataset_c_materials_project.csv` — **9,972 materials** matching perovskite
  stoichiometry (5-atom ABX₃ or 10-atom A₂BB′X₆), with band gap, is_metal,
  energy_per_atom, and symmetry.
- **Skew:** 5,761 / 9,972 (58%) metallic with `Eg=0`.
- **Filtering:** composition/stoichiometry only (3–6 elements, 5- or 10-atom
  formulas); **no `e_above_hull` stability cutoff** was applied. See the paper
  appendix for the full extraction protocol.

## Data release contents

The planned Zenodo archive contains, subject to upstream redistribution terms:

- Raw source files for public datasets A-C only.
- Processed feature matrices, scalers, metadata, and fixed splits.
- A manifest recording source URLs, retrieval dates, row counts, and checksums.
- The repository commit used to generate the processed artifacts.

Datasets A and E, the optional Dataset B variant, model checkpoints, and
exploratory outputs are not part of the reviewer release.

Materials Project data should retain its source attribution and retrieval
metadata. Third-party datasets should retain their original licenses and
citations rather than being presented as newly collected data.

## Preparation

After downloading the raw files, run the standard pipeline to build processed
features and stratified splits:

```bash
python3 src/data/prepare.py --dataset dataset_b
python3 src/data/prepare.py --dataset dataset_c
python3 src/data/prepare.py --dataset dataset_d
```

Outputs are written to `data/processed/{dataset}/`. These generated files are
not committed to Git; they belong in the versioned data release after the
release manifest has been finalized.
