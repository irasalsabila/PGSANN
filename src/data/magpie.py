"""Magpie feature generation matching Chenebuah B/C schema.

Computes the 54 Magpie features (27 elemental properties x mean/std) exactly as
used in Datasets B (oqmd_data.csv) and C (combine.csv). Elemental property
values are taken from mendeleev; composition-weighted mean and standard
deviation are computed over the elements present in the formula.

These match the column names in B/C:
    {prop}_mean, {prop}_std  for prop in the 27 Magpie properties.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

# 27 Magpie properties, matching the order/names in B/C columns.
MAGPIE_PROPS = [
    "z", "grp", "row", "atom_mass", "atom_rad", "av_ionrad", "av_rsp",
    "bp", "cal_atom_rad", "cov_rad", "ea", "heat_fus", "heat_vap", "ie",
    "iupac", "k", "mn", "mol_vol", "mp", "p", "pet_mn", "rho", "rsp",
    "spec_heat", "val", "vdw", "x",
]


@lru_cache(maxsize=None)
def _element_magpie(symbol: str):
    """Return dict of Magpie property values for an element via mendeleev."""
    import mendeleev

    try:
        el = mendeleev.element(symbol)
    except Exception:
        return {p: None for p in MAGPIE_PROPS}

    def _safe(getter):
        try:
            v = getter()
            return v
        except Exception:
            return None

    en = _safe(lambda: el.electronegativity() if callable(el.electronegativity)
               else el.electronegativity)
    ionic = _safe(lambda: el.ionic_radii)
    # Pick a representative ionic radius: prefer a common oxidation state.
    av_ionrad = None
    if ionic is not None:
        try:
            items = list(ionic)  # InstrumentedList of IonicRadius objects
            if items and hasattr(items[0], "ionic_radius"):
                # Prefer charge +2, then +1, +3, then any.
                for ch in (2, 1, 3, 4, 6):
                    for it in items:
                        if getattr(it, "charge", None) == ch and it.ionic_radius is not None:
                            av_ionrad = it.ionic_radius
                            break
                    if av_ionrad is not None:
                        break
                if av_ionrad is None:
                    av_ionrad = next((it.ionic_radius for it in items
                                      if it.ionic_radius is not None), None)
        except Exception:
            av_ionrad = None
    # Group/period may be None for some exotic elements.
    grp = _safe(lambda: el.group_id)
    period = _safe(lambda: el.period)
    ie0 = _safe(lambda: el.ionenergies.get(1) if el.ionenergies else None)
    # Magic number (Pettifor), thermal conductivity, polarizability.
    pet = _safe(lambda: el.pettifor_number)
    k = _safe(lambda: el.thermal_conductivity)
    polar = _safe(lambda: el.dipole_polarizability)
    vals = {
        "z": _safe(lambda: el.atomic_number),
        "grp": grp,
        "row": period,
        "atom_mass": _safe(lambda: el.atomic_weight),
        "atom_rad": _safe(lambda: el.atomic_radius),
        "av_ionrad": av_ionrad,
        "av_rsp": None,
        "bp": _safe(lambda: el.boiling_point),
        "cal_atom_rad": _safe(lambda: el.atomic_radius),
        "cov_rad": _safe(lambda: el.covalent_radius),
        "ea": _safe(lambda: el.electron_affinity),
        "heat_fus": _safe(lambda: el.fusion_heat),
        "heat_vap": _safe(lambda: el.evaporation_heat),
        "ie": ie0,
        "iupac": None,
        "k": k,
        "mn": _safe(lambda: el.melting_point),
        "mol_vol": _safe(lambda: el.atomic_volume),
        "mp": _safe(lambda: el.melting_point),
        "p": polar,
        "pet_mn": pet,
        "rho": _safe(lambda: el.density),
        "rsp": None,
        "spec_heat": _safe(lambda: el.specific_heat_capacity),
        "val": _safe(lambda: el.nvalence() if hasattr(el, "nvalence") else None),
        "vdw": _safe(lambda: el.vanderwaals_radius if hasattr(el, "vanderwaals_radius")
                     else el.covalent_radius),
        "x": en,
    }
    return vals


def _composition_weights(formula: str):
    """Return list of (symbol, fraction) for a formula (fraction = count/total)."""
    from src.data.features import parse_formula

    tokens = parse_formula(formula)
    total = sum(c for _, c in tokens)
    return [(sym, count / total) for sym, count in tokens]


def formula_magpie_features(formula: str):
    """Compute the 54 Magpie features (mean/std) for a formula.

    Properties unavailable in mendeleev (av_rsp, iupac, rsp) are filled with 0
    so the 54-column schema exactly matches Datasets B/C (constant columns are
    effectively unused by the model).
    """
    weights = _composition_weights(formula)
    out = {}
    for prop in MAGPIE_PROPS:
        vals = []
        for sym, frac in weights:
            v = _element_magpie(sym).get(prop)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            vals.append((v, frac))
        if not vals:
            # Unavailable property -> constant 0 (schema-consistent, unused).
            out[f"{prop}_mean"] = 0.0
            out[f"{prop}_std"] = 0.0
            continue
        arr = np.array([v for v, _ in vals])
        fracs = np.array([f for _, f in vals])
        # Composition-weighted mean and std (population).
        mean = np.sum(arr * fracs)
        var = np.sum(fracs * (arr - mean) ** 2)
        out[f"{prop}_mean"] = mean
        out[f"{prop}_std"] = np.sqrt(var)
    return out


def add_magpie_features(info) -> pd.DataFrame:
    """Return the df augmented with 54 Magpie columns (matching B/C schema).

    The input info must have a formula column. Returns a copy of the DataFrame
    (does NOT mutate info.df). Callers assign to info.df and update feature_cols.
    """
    from tqdm import tqdm

    df = info.df.copy()
    if not info.formula_col or info.formula_col not in df.columns:
        return df
    if f"{MAGPIE_PROPS[0]}_mean" in df.columns:
        return df  # already present
    print(f"    computing Magpie features for {len(df)} formulas ...", flush=True)
    rows = [
        formula_magpie_features(f)
        for f in tqdm(df[info.formula_col], desc="  Magpie", ncols=90, leave=False)
    ]
    magpie_df = pd.DataFrame(rows, index=df.index)
    df = pd.concat([df, magpie_df], axis=1)
    return df
