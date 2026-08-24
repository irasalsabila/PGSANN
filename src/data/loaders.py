import os
from dataclasses import dataclass, field
from typing import List

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

DATASET_A_CSV = os.path.join(
    DATA_DIR,
    "dataset_a",
    "lead_free_band_gap",
    "LF_final_ABX3+A2BbX6.csv",
)
DATASET_B_CSV = os.path.join(DATA_DIR, "dataset_b", "oqmd_data.csv")
DATASET_B_VARIANT_CSV = os.path.join(DATA_DIR, "dataset_b", "abc3_data.csv")
DATASET_C_CSV = os.path.join(DATA_DIR, "dataset_c", "combine.csv")
DATASET_E_CSV = os.path.join(DATA_DIR, "dataset_e", "double_perovskites_gap.csv")
DATASET_D_CSV = os.path.join(DATA_DIR, "dataset_d", "mp_perovskites.csv")


@dataclass
class DatasetInfo:
    name: str
    path: str
    df: pd.DataFrame = field(repr=False)
    feature_cols: List[str] = field(default_factory=list)
    target_col: str = ""
    meta_cols: List[str] = field(default_factory=list)
    formula_col: str = ""

    @property
    def X(self) -> pd.DataFrame:
        return self.df[self.feature_cols]

    @property
    def y(self) -> pd.Series:
        return self.df[self.target_col]


def load_dataset_a() -> DatasetInfo:
    """Load Moeinimajd & Samadpour lead-free perovskite dataset (1016 rows)."""
    df = pd.read_csv(DATASET_A_CSV)
    df = df.rename(columns={"structure-type": "structure_type"})
    formula_col = "pretty_formula"
    target_col = "band_gap"
    # structure_type is a binary structural index and counts as one of the
    # 55 features (PRD); keep it in the feature set, only formula is metadata.
    meta_cols = ["pretty_formula"]
    feature_cols = [c for c in df.columns if c not in meta_cols + [target_col]]
    return DatasetInfo(
        name="dataset_a",
        path=DATASET_A_CSV,
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        meta_cols=meta_cols,
        formula_col=formula_col,
    )


def load_dataset_b() -> DatasetInfo:
    """Load Chenebuah/OQMD ABX3 dataset (16323 rows, Magpie descriptors)."""
    df = pd.read_csv(DATASET_B_CSV)
    formula_col = "name"
    target_col = "Eg"
    # Drop IDs / provenance; keep descriptors, gtf, of as features.
    meta_cols = ["name", "entry_id", "icsd_id"]
    # sg (space group), cs/cs1 (crystal system) are categorical structural cols.
    structural_categorical = ["sg", "cs", "cs1"]
    feature_cols = [
        c for c in df.columns if c not in meta_cols + structural_categorical + [target_col]
    ]
    return DatasetInfo(
        name="dataset_b",
        path=DATASET_B_CSV,
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        meta_cols=meta_cols,
        formula_col=formula_col,
    )


def load_dataset_b_variant() -> DatasetInfo:
    """Load abc3_data.csv (4557 rows, lattice + MP targets)."""
    df = pd.read_csv(DATASET_B_VARIANT_CSV)
    formula_col = "formula"
    target_col = "band_gap"
    meta_cols = ["formula", "mp_id"]
    if "mp_id" not in df.columns:
        meta_cols = ["formula"]
    feature_cols = [c for c in df.columns if c not in meta_cols + [target_col]]
    return DatasetInfo(
        name="dataset_b_variant",
        path=DATASET_B_VARIANT_CSV,
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        meta_cols=meta_cols,
        formula_col=formula_col,
    )


def load_dataset_c() -> DatasetInfo:
    """Load Chenebuah combine.csv (1453 rows, Magpie descriptors, target Eg)."""
    df = pd.read_csv(DATASET_C_CSV)
    formula_col = "formula"
    target_col = "Eg"
    meta_cols = ["formula"]
    # gtf/of are tolerance/octahedral factors; keep as features (physics).
    feature_cols = [c for c in df.columns if c not in meta_cols + [target_col]]
    return DatasetInfo(
        name="dataset_c",
        path=DATASET_C_CSV,
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        meta_cols=meta_cols,
        formula_col=formula_col,
    )


def load_dataset_combined() -> DatasetInfo:
    """Deduplicated combination of Datasets B (OQMD) and C (combine).

    Both are Chenebuah Magpie-descriptor datasets sharing the target `Eg`.
    Merge rule (Option B):
      - Formulas only in B  -> keep B's row.
      - Formulas only in C  -> keep C's row.
      - Formulas in both with consistent Eg (|dEg| <= tol) -> keep B's row (one entry).
      - Formulas in both with conflicting Eg (> tol)       -> DROPPED here (they
        stay separated in the individual B and C datasets; see plans/02).
    The merged frame uses only the common Magpie feature columns + Eg + t/mu.
    """
    b = pd.read_csv(DATASET_B_CSV).rename(columns={"name": "formula"})
    c = pd.read_csv(DATASET_C_CSV)
    tol = 0.01  # eV

    common_feats = sorted(set(b.columns) & set(c.columns) - {"formula", "Eg"})
    cols = ["formula", "Eg"] + common_feats

    b = b[cols].copy()
    c = c[cols].copy()

    b_form = set(b["formula"])
    c_form = set(c["formula"])
    overlap = b_form & c_form

    # B-only rows
    b_only = b[~b["formula"].isin(overlap)].copy()
    # C-only rows
    c_only = c[~c["formula"].isin(overlap)].copy()

    # Overlapping rows: keep B's if Eg consistent, else drop.
    b_map = b.set_index("formula")["Eg"].to_dict()
    c_map = c.set_index("formula")["Eg"].to_dict()
    consistent = [f for f in overlap if abs(b_map[f] - c_map[f]) <= tol]
    conflicting = [f for f in overlap if f not in consistent]
    b_overlap_keep = b[b["formula"].isin(consistent)].copy()

    merged = pd.concat([b_only, c_only, b_overlap_keep], ignore_index=True)
    merged = merged.drop_duplicates(subset=["formula"]).reset_index(drop=True)

    # Report for transparency.
    print(f"[dataset_combined] B={len(b)} C={len(c)} "
          f"overlap={len(overlap)} consistent={len(consistent)} "
          f"conflicting_dropped={len(conflicting)} -> merged={len(merged)}", flush=True)

    formula_col = "formula"
    target_col = "Eg"
    meta_cols = ["formula"]
    feature_cols = [c for c in merged.columns if c not in meta_cols + [target_col]]
    return DatasetInfo(
        name="dataset_combined",
        path="B+C (dedup)",
        df=merged,
        feature_cols=feature_cols,
        target_col=target_col,
        meta_cols=meta_cols,
        formula_col=formula_col,
    )


def load_dataset_d() -> DatasetInfo:
    """Load Materials Project perovskites (from download_dataset_d.py).

    Requires data/raw/dataset_d/mp_perovskites.csv (run the downloader first).
    """
    if not os.path.exists(DATASET_D_CSV):
        raise FileNotFoundError(
            f"No {DATASET_D_CSV}. Run: "
            "export MP_API_KEY=... && python src/data/download_dataset_d.py"
        )
    df = pd.read_csv(DATASET_D_CSV)
    formula_col = "formula"
    target_col = "band_gap"
    meta_cols = ["material_id", "formula", "is_metal", "crystal_system"]
    # Keep only numeric descriptors + target.
    feature_cols = [c for c in df.columns
                    if c not in meta_cols + [target_col] and pd.api.types.is_numeric_dtype(df[c])]
    return DatasetInfo(
        name="dataset_d",
        path=DATASET_D_CSV,
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        meta_cols=meta_cols,
        formula_col=formula_col,
    )


def load_dataset_e() -> DatasetInfo:
    """Load FionaZhang double-perovskites gap dataset (1306 rows, matminer).

    Columns: formula, a_1, b_1, a_2, b_2, gap gllbsc. Chemistry features are
    derived from the formula (t, mu) in the feature-engineering step.
    """
    df = pd.read_csv(DATASET_E_CSV)
    df = df.rename(columns={"gap gllbsc": "band_gap"})
    formula_col = "formula"
    target_col = "band_gap"
    meta_cols = ["formula", "a_1", "b_1", "a_2", "b_2"]
    feature_cols = [c for c in df.columns if c not in meta_cols + [target_col]]
    return DatasetInfo(
        name="dataset_e",
        path=DATASET_E_CSV,
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        meta_cols=meta_cols,
        formula_col=formula_col,
    )


def load_dataset_f() -> DatasetInfo:
    """Load NOMAD perovskite solar-cells database.

    NOTE: Dataset F is a solar-cell DEVICE database (A/B/C ion molecular data
    for hybrid perovskite photovoltaics), not a clean band-gap regression table.
    It is not directly comparable to A-C; it requires separate preprocessing and
    is treated as supplementary rather than a band-gap benchmark.
    """
    raise NotImplementedError(
        "Dataset F (NOMAD perovskite solar cells) is a device database, not a "
        "band-gap regression set. It requires custom preprocessing and is not "
        "plugged into the standard pipeline. See plans/02."
    )


def load_dataset(name: str) -> DatasetInfo:
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name in ("a", "dataset_a", "dataset a", "moeinimajd", "lead_free"):
        return load_dataset_a()
    if name in ("b", "dataset_b", "dataset b", "suhendar", "oqmd", "chenebuah"):
        return load_dataset_b()
    if name in ("b_variant", "abc3", "dataset_b_variant", "lattice"):
        return load_dataset_b_variant()
    if name in ("c", "dataset_c", "dataset c", "combine", "chenebuah_combine"):
        return load_dataset_c()
    if name in ("combined", "dataset_combined", "bc", "dataset_bc", "b_c",
                "chenebuah_combined", "b+c"):
        return load_dataset_combined()
    if name in ("d", "dataset_d", "dataset d", "materials_project", "mp", "mp_api"):
        return load_dataset_d()
    if name in ("e", "dataset_e", "dataset e", "fionazhang", "double_perovskite",
                "double_perovskites_gap", "double_perovskites"):
        return load_dataset_e()
    if name in ("f", "dataset_f", "dataset f", "nomad", "perovskite_solar_cells"):
        return load_dataset_f()
    raise ValueError(f"Unknown dataset: {name}")
