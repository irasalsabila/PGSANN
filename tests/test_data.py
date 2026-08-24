import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    compute_radii,
    octahedral_factor,
    parse_formula,
    site_elements,
    tolerance_factor,
)
from src.data.loaders import load_dataset_a, load_dataset_b, load_dataset_c, load_dataset_combined, load_dataset_d, load_dataset_e
from src.data.prepare import load_splits, prepare_dataset


def test_dataset_a_shape_and_features():
    a = load_dataset_a()
    assert len(a.df) == 1016
    assert len(a.feature_cols) == 55
    assert a.target_col == "band_gap"
    assert not a.X.isna().any().any()


def test_dataset_b_shape():
    b = load_dataset_b()
    assert len(b.df) == 16323
    assert b.target_col == "Eg"


def test_dataset_c_shape():
    c = load_dataset_c()
    assert len(c.df) == 1453
    assert c.target_col == "Eg"
    assert len(c.feature_cols) >= 50  # Magpie descriptors


def test_dataset_combined_dedup():
    """B+C combined must have no duplicate formulas and drop conflicting overlaps."""
    d = load_dataset_combined()
    # merged size < B + C (dedup happened)
    assert 0 < len(d.df) < 16323 + 1453
    # no duplicate formulas
    assert d.df["formula"].duplicated().sum() == 0
    # target present
    assert d.target_col == "Eg"
    assert len(d.feature_cols) >= 50


def test_dataset_e_shape_and_features():
    e = load_dataset_e()
    assert len(e.df) == 1306
    assert e.target_col == "band_gap"
    # add_elemental_descriptors should add en/mass/radius/valence
    from src.data.features import add_elemental_descriptors, add_physics_features

    e2 = add_physics_features(add_elemental_descriptors(e))
    assert any("en_mean" in c for c in e2.feature_cols)
    assert "t" in e2.feature_cols
    assert not e2.df[["t", "mu"]].isna().all().any()


def test_dataset_d_shape():
    import os

    from src.data.loaders import DATASET_D_CSV

    if not os.path.exists(DATASET_D_CSV):
        pytest.skip("Dataset D not downloaded (run download_dataset_d.py)")
    d = load_dataset_d()
    assert d.target_col == "band_gap"
    assert len(d.df) > 1000  # large skew-heavy set


def test_parse_formula_organic_cation():
    # MA, FA, DMA must stay single tokens; digits grouped correctly.
    assert parse_formula("MAPbI3") == [("MA", 1), ("Pb", 1), ("I", 3)]
    assert parse_formula("Cs2AgBiCl6") == [
        ("Cs", 2), ("Ag", 1), ("Bi", 1), ("Cl", 6),
    ]


def test_site_elements():
    a, b, x = site_elements("CsSnI3")
    assert a == ["Cs"]
    assert b == ["Sn"]
    assert x == ["I"]


def test_tolerance_factor_known():
    # CsSnI3: r_Cs=1.88, r_Sn=0.69, r_I=2.20 -> t ~ 0.998
    rA, rB, rX = compute_radii("CsSnI3")
    t = tolerance_factor(rA, rB, rX)
    assert t == pytest.approx(0.998, abs=0.01)
    mu = octahedral_factor(rB, rX)
    assert mu == pytest.approx(0.314, abs=0.01)


def test_tolerance_stable_range():
    # Stable 3D perovskites: 0.8 <= t <= 1.05. CsSnI3 should be inside.
    rA, rB, rX = compute_radii("CsSnI3")
    t = tolerance_factor(rA, rB, rX)
    assert 0.8 <= t <= 1.05


def test_physics_features_no_nan():
    from src.data.features import add_physics_features

    a = add_physics_features(load_dataset_a())
    assert "t" in a.df.columns
    assert "mu" in a.df.columns
    assert a.df["t"].notna().all()
    assert a.df["mu"].notna().all()


def test_prepare_and_splits():
    meta = prepare_dataset("dataset_a")
    assert meta["n_train"] + meta["n_val"] + meta["n_test"] == 1016
    s = load_splits("dataset_a")
    assert s["X_train"].shape[1] == len(meta["feature_cols"])
    assert len(s["y_train"]) == meta["n_train"]
