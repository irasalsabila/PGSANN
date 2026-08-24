import math
from functools import lru_cache

import numpy as np
import pandas as pd

from src.data.loaders import DatasetInfo

# Ionic radii (Angstrom) by site, using Shannon ionic radii.
# A-site: 12-coordinate cations; B-site: 6-coordinate; X-site: 6-coordinate anions.
RADII_A = {
    "Cs": 1.88, "Rb": 1.72, "K": 1.64, "Na": 1.39, "Li": 0.76,
    "Ba": 1.61, "Sr": 1.44, "Ca": 1.34, "La": 1.36, "Ce": 1.34,
    "Pr": 1.32, "Nd": 1.27, "Sm": 1.24, "Eu": 1.30, "Gd": 1.22,
    "Tb": 1.18, "Dy": 1.17, "Ho": 1.15, "Er": 1.14, "Tm": 1.13,
    "Bi": 1.31, "Sn": 1.12, "Pb": 1.29,
    # organic A-site cations (approximate effective ionic radius)
    "MA": 2.17, "FA": 2.53, "DMA": 2.72, "EA": 2.75, "GA": 2.46,
}
RADII_B = {
    "Mg": 0.72, "Ca": 1.00, "Sr": 1.18, "Ba": 1.35, "Ti": 0.605,
    "Zr": 0.72, "Hf": 0.71, "V": 0.54, "Nb": 0.64, "Ta": 0.64,
    "Cr": 0.615, "Mo": 0.59, "W": 0.60, "Mn": 0.645, "Fe": 0.645,
    "Co": 0.545, "Ni": 0.69, "Cu": 0.73, "Zn": 0.74, "Cd": 0.95,
    "Al": 0.535, "Ga": 0.62, "In": 0.80, "Tl": 0.885, "Ge": 0.53,
    "Sn": 0.69, "Pb": 1.19, "Sb": 0.76, "Bi": 0.76, "Sc": 0.745,
    "Y": 0.90, "La": 1.032, "Ce": 1.01, "Pr": 0.99, "Nd": 0.983,
    "Sm": 0.958, "Eu": 0.947, "Gd": 0.938, "Tb": 0.923, "Dy": 0.912,
    "Ho": 0.901, "Er": 0.890, "Tm": 0.880, "Yb": 0.868, "Lu": 0.861,
    "Ru": 0.62, "Rh": 0.665, "Pd": 0.86, "Ag": 1.15, "Ir": 0.625,
    "Pt": 0.80, "Au": 0.85, "U": 0.89, "Th": 1.05, "Np": 0.87,
    "As": 0.58, "Sb": 0.76, "P": 0.44,
}
RADII_X = {
    "F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20, "O": 1.40,
    "S": 1.84, "Se": 1.98, "Te": 2.21, "N": 1.46,
}

# Site-radius lookup; falls back to B-site value if a symbol is missing on its site.
_SITE_RADII = {"A": RADII_A, "B": RADII_B, "X": RADII_X}

# All recognized element/organic-cation symbols, for longest-match tokenization.
ELEMENT_SYMBOLS = (
    set(RADII_A) | set(RADII_B) | set(RADII_X)
    | {
        "H", "He", "Be", "B", "C", "Ne", "Si", "P", "As",
        "Kr", "Te", "Yb", "Lu", "Re", "Os",
    }
)


def _radius(symbol: str, site: str) -> float:
    radii = _SITE_RADII.get(site, RADII_B)
    # Fall back to B-site radii for unknown symbols on A-site, etc.
    if symbol not in radii and site != "B":
        radii = RADII_B
    return radii.get(symbol, np.nan)


def _nanmean_or(values, default):
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return default
    return float(np.mean(values))


def parse_formula(formula: str):
    """Very small formula parser: returns list of (symbol, count) tokens.
    Handles simple formulas like CsSnI3, Cs2SnI6, Cs2AgBiCl6.
    Organic cations (MA, FA, DMA, EA, GA) are single-element tokens.
    """
    formula = formula.replace(" ", "")
    # Insert separators between element boundaries and numbers.
    # Use a known-symbol dictionary with longest-match-first so organic
    # cations (MA, FA, DMA) and single-letter elements (S, I, N) parse correctly.
    import re

    symbols = sorted(ELEMENT_SYMBOLS, key=len, reverse=True)
    tokens = []
    i = 0
    n = len(formula)
    while i < n:
        if formula[i].isdigit():
            j = i
            while j < n and formula[j].isdigit():
                j += 1
            tokens.append(formula[i:j])
            i = j
            continue
        matched = False
        for sym in symbols:
            if formula.startswith(sym, i):
                tokens.append(sym)
                i += len(sym)
                matched = True
                break
        if not matched:
            i += 1
    # Simple grouping: read element then optional number.
    parsed = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.isdigit():
            i += 1
            continue
        count = 1
        if i + 1 < len(tokens) and tokens[i + 1].isdigit():
            count = int(tokens[i + 1])
            i += 2
        else:
            i += 1
        parsed.append((tok, count))
    return parsed


def site_elements(formula: str):
    """Return (a_species, b_species, x_species) element lists for ABX3 / A2B'B''X6.
    Approximate heuristic based on electronegativity ordering (B more covalent /
    intermediate electronegativity). For well-known perovskite stoichiometries we
    classify by structure: A2B'B''X6 -> two B; ABX3 -> one B.
    """
    tokens = parse_formula(formula)
    total = sum(c for _, c in tokens)
    # Identify X: highest electronegativity (halides O S Se N Te).
    x_elements = [
        (s, c) for s, c in tokens
        if s in {"F", "Cl", "Br", "I", "O", "S", "Se", "Te", "N"}
    ]
    non_x = [(s, c) for s, c in tokens if s not in {x for x, _ in x_elements}]
    # B-site: typically metallic / semimetallic elements (intermediate electronegativity).
    # Heuristic: A-site are large alkali/alkaline/organic; B-site are transition/post-transition metals.
    b_set = {
        "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn", "Fe",
        "Co", "Ni", "Cu", "Zn", "Cd", "Al", "Ga", "In", "Tl", "Ge", "Sn",
        "Pb", "Sb", "Bi", "Sc", "Y", "Ru", "Rh", "Pd", "Ag", "Ir", "Pt",
        "Au", "U", "Th", "Np", "Mg", "Ca", "Sr", "Ba", "La", "Ce", "Pr",
        "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    }
    a_set = {"Cs", "Rb", "K", "Na", "Li", "MA", "FA", "DMA", "EA", "GA"}
    a_species = [s for s, _ in non_x if s in a_set]
    b_species = [s for s, _ in non_x if s in b_set]
    # Fallback: assign remaining non-X, non-A elements to B-site.
    for s, c in non_x:
        if s not in a_set and s not in b_set:
            b_species.append(s)
    x_species = [s for s, _ in x_elements]
    return a_species, b_species, x_species


def compute_radii(formula: str):
    """Compute effective r_A, r_B, r_X from a formula.
    For multiple species on a site, returns composition-weighted average.
    """
    a_species, b_species, x_species = site_elements(formula)
    a_species = a_species or ["Cs"]
    b_species = b_species or ["Ti"]
    x_species = x_species or ["Br"]
    r_A = _nanmean_or([_radius(s, "A") for s in a_species], _radius("Cs", "A"))
    r_B = _nanmean_or([_radius(s, "B") for s in b_species], _radius("Ti", "B"))
    r_X = _nanmean_or([_radius(s, "X") for s in x_species], _radius("Br", "X"))
    return r_A, r_B, r_X


def tolerance_factor(r_A, r_B, r_X) -> float:
    """Goldschmidt tolerance factor t = (r_A + r_X) / (sqrt(2) * (r_B + r_X))."""
    denom = math.sqrt(2.0) * (r_B + r_X)
    if denom == 0 or any(math.isnan(x) for x in (r_A, r_B, r_X)):
        return np.nan
    return (r_A + r_X) / denom


def octahedral_factor(r_B, r_X) -> float:
    """Octahedral factor mu = r_B / r_X."""
    if r_X == 0 or any(math.isnan(x) for x in (r_B, r_X)):
        return np.nan
    return r_B / r_X


def add_physics_features(info: DatasetInfo) -> DatasetInfo:
    """Add t (tolerance factor) and mu (octahedral factor) columns.
    Dataset B already provides gtf/of; we still recompute for consistency where
    formulas are available, but fall back to provided gtf/of when formula parsing
    is unreliable.
    """
    df = info.df.copy()
    has_provided = "gtf" in df.columns and "of" in df.columns
    if has_provided:
        # Dataset B provides authoritative gtf (tolerance) and of (octahedral);
        # prefer these over heuristic formula-derived values.
        df["t"] = df["gtf"]
        df["mu"] = df["of"]
    if info.formula_col and info.formula_col in df.columns:
        radii = df[info.formula_col].map(compute_radii)
        df["r_A"] = radii.map(lambda r: r[0])
        df["r_B"] = radii.map(lambda r: r[1])
        df["r_X"] = radii.map(lambda r: r[2])
        if not has_provided:
            df["t"] = df.apply(
                lambda row: tolerance_factor(row["r_A"], row["r_B"], row["r_X"]), axis=1
            )
            df["mu"] = df.apply(
                lambda row: octahedral_factor(row["r_B"], row["r_X"]), axis=1
            )
    else:
        # No formula: use provided tolerance/octahedral columns if present.
        if "gtf" in df.columns:
            df["t"] = df["gtf"]
        if "of" in df.columns:
            df["mu"] = df["of"]

    info.df = df
    # Add t, mu (and radii) to feature set if not already present.
    for col in ["t", "mu"]:
        if col in df.columns and col not in info.feature_cols:
            info.feature_cols = info.feature_cols + [col]
    return info


@lru_cache(maxsize=None)
def _element_descriptors(symbol: str):
    """Return (electronegativity, atomic_mass, covalent_radius, valence) via mendeleev."""
    import mendeleev
    try:
        el = mendeleev.element(symbol)
        en = el.electronegativity() if callable(el.electronegativity) else el.electronegativity
        radius = el.covalent_radius
        mass = el.atomic_weight
        # crude valence proxy: electrons
        elec = el.electrons
        return (float(en) if en is not None else np.nan,
                float(mass) if mass is not None else np.nan,
                float(radius) if radius is not None else np.nan,
                float(elec) if elec is not None else np.nan)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)


def formula_elemental_descriptors(formula: str):
    """Composition-averaged elemental descriptors for a formula.

    Returns (en_mean, mass_mean, radius_mean, valence_mean) over all elements in
    the formula (weighted by elemental stoichiometry). Used for formula-only
    datasets such as Dataset E (double perovskites).
    """
    tokens = parse_formula(formula)
    total = sum(c for _, c in tokens)
    sums = [0.0, 0.0, 0.0, 0.0]
    counts = 0
    for sym, count in tokens:
        d = _element_descriptors(sym)
        if np.isnan(d[0]):
            continue
        w = count / total
        for i in range(4):
            sums[i] += w * d[i]
        counts += 1
    if counts == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    return tuple(sums)


def add_elemental_descriptors(info: DatasetInfo) -> DatasetInfo:
    """Add composition-averaged elemental descriptors (EN, mass, radius, valence)
    for formula-only datasets (e.g., Dataset E). Returns updated info."""
    if not info.formula_col or info.formula_col not in info.df.columns:
        return info
    cols = ["en_mean", "mass_mean", "rad_mean", "valence_mean"]
    # Skip if already added.
    if cols[0] in info.df.columns:
        return info
    desc = info.df[info.formula_col].map(formula_elemental_descriptors)
    for i, c in enumerate(cols):
        info.df[c] = desc.map(lambda r: r[i])
    info.feature_cols = info.feature_cols + cols
    return info
