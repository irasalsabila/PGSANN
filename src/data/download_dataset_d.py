"""Download Dataset D: Materials Project perovskites (band gap + properties).

Usage:
    # set your API key (from https://next-gen.materialsproject.org/api)
    export MP_API_KEY=your_key_here
    python src/data/download_dataset_d.py

Outputs:
    data/raw/dataset_d/mp_perovskites.csv

Strategy: query Materials Project Summary for materials with 3-6 elements
(perovskites are ternary/quaternary ABX3 or A2BB'X6), then keep only those whose
composition has 5 atoms (ABX3) or 10 atoms (A2BB'X6). This yields a large,
skew-heavy dataset (many metals have band_gap=0), which is the challenge PG-SANN
addresses.

The full MP summary query can take ~1 min (60k+ materials). If you want to keep
BOTH the filtered perovskites AND the raw all-materials pull, set --save-all.

NOTE: The API key is read from MP_API_KEY env var (never hard-coded). Pass
--api-key as a fallback but do not commit it.
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=None, help="MP API key (prefer MP_API_KEY env)")
    ap.add_argument("--save-all", action="store_true",
                    help="also save the full unfiltered query to mp_all.csv")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        raise SystemExit(
            "No API key. Set MP_API_KEY env var or pass --api-key. "
            "Get a key at https://next-gen.materialsproject.org/api"
        )

    from mp_api.client import MPRester

    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "data", "raw", "dataset_d")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "mp_perovskites.csv")

    print("Querying Materials Project Summary ...", flush=True)
    with MPRester(api_key) as mpr:
        results = mpr.materials.summary.search(
            num_elements=[3, 4, 5, 6],
            fields=[
                "material_id", "formula_pretty", "composition",
                "band_gap", "is_metal", "energy_per_atom",
                "symmetry", "nsites",
            ],
        )

    rows = []
    for item in results:
        comp = getattr(item, "composition", None)
        n_atoms = sum(comp.values()) if comp else 0
        # Perovskite stoichiometry: ABX3 = 5 atoms, A2BB'X6 = 10 atoms.
        if n_atoms not in (5, 10):
            continue
        sym = getattr(item, "symmetry", None)
        rows.append({
            "material_id": item.material_id,
            "formula": item.formula_pretty,
            "band_gap": getattr(item, "band_gap", None),
            "is_metal": getattr(item, "is_metal", None),
            "energy_per_atom": getattr(item, "energy_per_atom", None),
            "nsites": n_atoms,
            "crystal_system": getattr(sym, "crystal_system", None) if sym else None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nPerovskite filter: {len(df)} materials (ABX3/A2BB'X6) -> {out}", flush=True)
    if len(df):
        print(f"  zeros (metallic, band_gap=0): {(df['band_gap'].fillna(0) == 0).sum()} "
              f"({100*(df['band_gap'].fillna(0)==0).mean():.0f}%)", flush=True)
        print(f"  band_gap range: [{df['band_gap'].min():.3f}, {df['band_gap'].max():.3f}]", flush=True)
        print("  sample:", df['formula'].head(5).tolist(), flush=True)

    if args.save_all:
        all_out = os.path.join(out_dir, "mp_all.csv")
        pd.DataFrame([
            {"material_id": r.material_id, "formula": r.formula_pretty,
             "band_gap": getattr(r, "band_gap", None),
             "is_metal": getattr(r, "is_metal", None)}
            for r in results
        ]).to_csv(all_out, index=False)
        print(f"Saved full query -> {all_out} ({len(results)} rows)", flush=True)


if __name__ == "__main__":
    main()
