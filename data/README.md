# Data

Raw and processed data are **not** distributed in this repository (see
`.gitignore`). This README documents the three benchmark datasets used in the
paper (labeled A, B, C in the manuscript) and how to obtain them.

## Dataset A — OQMD ABX₃ perovskites (OQMD, Chenebuah)

- **Source:** `github.com/chenebuah/ML_abx3_dataset` (arXiv:2312.11335)
  and `github.com/chenebuah/perovskite-ML` (Mater. Today Commun. 27, 102462, 2021)
- **File:** `oqmd_data.csv` — **16,323 rows**, Magpie statistical descriptors
  + lattice parameters; target `Eg`.
- **Skew:** 11,316 of 16,323 entries are metallic (`Eg=0`) → heavy zero-gap skew,
  the central challenge PG-SANN addresses.
- **Note:** the manuscript's "45,570 compounds" figure refers to the full raw
  OQMD extraction; the public repos distribute the reproducible 16,323-row file.

## Dataset B — Chenebuah combined data

- **Source:** `github.com/chenebuah/perovskite-ML` (`combine.zip`)
- **File:** `combine.csv` — **1,453 rows**, Magpie descriptors + `gtf`, `of`,
  `rho`, `Ehull`, `Ef`; target `Eg`.

## Dataset C — Materials Project perovskites

- **Source:** Materials Project API (`next-gen.materialsproject.org`)
- **Download (API key required):**
  ```bash
  export MP_API_KEY=your_key   # from https://next-gen.materialsproject.org/api
  python3 src/data/download_dataset_d.py
  ```
- **File:** `mp_perovskites.csv` — **9,972 materials** matching perovskite
  stoichiometry (5-atom ABX₃ or 10-atom A₂BB′X₆), with band gap, is_metal,
  energy_per_atom, and symmetry.
- **Skew:** 5,761 / 9,972 (58%) metallic with `Eg=0`.
- **Filtering:** composition/stoichiometry only (3–6 elements, 5- or 10-atom
  formulas); **no `e_above_hull` stability cutoff** was applied. See the paper
  appendix for the full extraction protocol.

## Preparation

After downloading the raw files, run the standard pipeline to build processed
features and stratified splits:

```bash
python3 src/data/prepare.py --dataset dataset_b
python3 src/data/prepare.py --dataset dataset_c
python3 src/data/prepare.py --dataset dataset_d
```

Outputs are written to `data/processed/{dataset}/` and `data/splits/`
(not committed).
