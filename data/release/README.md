# Zenodo Release Package

This directory contains only the three benchmark datasets used in the final
reviewer release. Upload the three ZIP archives to Zenodo; the uncompressed
folders are retained locally for inspection.

| Folder | Contents |
|---|---|
| `dataset_a_oqmd/` | OQMD ABX3 raw table and processed splits |
| `dataset_b_chenebuah/` | Chenebuah combined raw table and processed splits |
| `dataset_c_materials_project/` | Materials Project raw table and processed splits |

Upload these files:

```text
dataset_a_oqmd.zip
dataset_b_chenebuah.zip
dataset_c_materials_project.zip
```

Each folder contains `raw/` and `processed/` subdirectories. The processed
directory contains the feature tables, targets, NumPy train/validation/test
arrays, fitted scaler, and `meta.json`.

The public release names differ from the internal pipeline identifiers:
`dataset_a_oqmd` is generated from internal `dataset_b`, `dataset_b_chenebuah`
from `dataset_c`, and `dataset_c_materials_project` from `dataset_d`.
