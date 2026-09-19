#!/usr/bin/env python3
"""
01_prepare_labels.py
--------------------
Phase 0 for the WristTrauma-AI custom-CNN pivot.

Reads the existing YOLO box labels + the cleaned metadata CSV and produces a
per-image MULTI-LABEL table over 4 MERGED clinical classes:

    0 fracture              <- yolo 3
    1 foreign_material      <- yolo 4 (metal)      + yolo 2 (foreignbody)
    2 softtissue_indirect   <- yolo 7 (softtissue) + yolo 6 (pronatorsign) + yolo 5 (periostealreaction)
    3 bone_lesion           <- yolo 0 (boneanomaly)+ yolo 1 (bonelesion)

    yolo class 8 (text) is DROPPED.

An image with none of the above becomes an all-zero row = NORMAL (no findings).

Output:  data/classification/labels_multilabel.csv

This script is READ-ONLY with respect to the dataset. It only writes the new CSV.
Run it from anywhere:  python scripts/01_prepare_labels.py
"""

from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError:
    sys.exit("[ERROR] pandas is not installed in this environment. Run:  pip install pandas")

# ---------------------------------------------------------------------------
# CONFIG  -- only edit PROJECT_ROOT if you move this script out of scripts/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]              # scripts/ -> repo root
META_CSV     = PROJECT_ROOT / "data" / "dataset_metadata_cleaned.csv"
LABELS_DIR   = PROJECT_ROOT / "data" / "processed" / "labels"   # contains train/ val/ test/
OUT_CSV      = PROJECT_ROOT / "data" / "classification" / "labels_multilabel.csv"

# original yolo class id -> merged class index  (None means drop)
YOLO_TO_MERGED = {
    3: 0,                 # fracture
    4: 1, 2: 1,           # foreign_material
    7: 2, 6: 2, 5: 2,     # softtissue_indirect
    0: 3, 1: 3,           # bone_lesion
    8: None,              # text -> dropped
}
MERGED_NAMES = ["fracture", "foreign_material", "softtissue_indirect", "bone_lesion"]
N_CLASSES = len(MERGED_NAMES)

# rough expected totals (from the earlier repo audit) -- just a sanity anchor
EXPECTED = {"fracture": 13550, "foreign_material": 715,
            "softtissue_indirect": 3240, "bone_lesion": 234}


# ---------------------------------------------------------------------------
def find_label_file(stem: str, split) -> Path | None:
    """Locate the .txt label for an image, tolerant of split-folder naming."""
    guess = LABELS_DIR / str(split) / f"{stem}.txt"
    if guess.exists():
        return guess
    if LABELS_DIR.exists():
        for sub in LABELS_DIR.iterdir():
            if sub.is_dir():
                cand = sub / f"{stem}.txt"
                if cand.exists():
                    return cand
    return None


def multihot_for_label(path: Path | None):
    """Return (multi-hot vector over merged classes, found_flag)."""
    vec = [0] * N_CLASSES
    if path is None:
        return vec, False
    try:
        lines = path.read_text().strip().splitlines()
    except Exception:
        return vec, False
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            cid = int(float(ln.split()[0]))
        except (ValueError, IndexError):
            continue
        merged = YOLO_TO_MERGED.get(cid, None)
        if merged is not None:
            vec[merged] = 1
    return vec, True


def pick(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
def main():
    if not META_CSV.exists():
        sys.exit(f"[ERROR] metadata csv not found: {META_CSV}")
    if not LABELS_DIR.exists():
        sys.exit(f"[ERROR] labels dir not found: {LABELS_DIR}")

    df = pd.read_csv(META_CSV)
    print(f"[info] loaded metadata rows: {len(df)}")
    print(f"[info] columns: {list(df.columns)}")

    col_stem = pick(df, "filestem", "file_stem", "filename", "image")
    col_split = pick(df, "split")
    col_age  = pick(df, "age")
    col_sex  = pick(df, "gender", "sex")
    col_side = pick(df, "laterality", "side")
    col_proj = pick(df, "projection", "view")

    if col_stem is None or col_split is None:
        sys.exit(f"[ERROR] need a filestem column and a split column. Found: {list(df.columns)}")

    print(f"[info] split value counts:\n{df[col_split].value_counts(dropna=False)}\n")

    rows, missing = [], 0
    for _, r in df.iterrows():
        stem = str(r[col_stem])
        split = r[col_split]
        lf = find_label_file(stem, split)
        vec, found = multihot_for_label(lf)
        if not found:
            missing += 1
        rows.append({
            "filestem": stem,
            "split": split,
            **{f"has_{MERGED_NAMES[i]}": vec[i] for i in range(N_CLASSES)},
            "n_findings": sum(vec),
            "is_abnormal": int(sum(vec) > 0),
            "age": r[col_age] if col_age else "",
            "sex": r[col_sex] if col_sex else "",
            "side": r[col_side] if col_side else "",
            "projection": r[col_proj] if col_proj else "",
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[info] wrote {OUT_CSV}   ({len(out)} rows)")
    if missing:
        print(f"[warn] {missing} images had NO label file found -> treated as NORMAL (no findings)")

    # ---- per-split summary (what we actually care about) ----
    print("\n=== per-split positive-image counts per merged class ===")
    for sp, g in out.groupby("split"):
        n = len(g)
        print(f"\n[{sp}]  images = {n}")
        for name in MERGED_NAMES:
            pos = int(g[f"has_{name}"].sum())
            print(f"    {name:22s}: {pos:6d}  ({100*pos/n:5.1f}%)")
        normal = int((g["is_abnormal"] == 0).sum())
        print(f"    {'NORMAL (no findings)':22s}: {normal:6d}  ({100*normal/n:5.1f}%)")

    # ---- whole-dataset sanity check vs known audit numbers ----
    print("\n=== whole-dataset totals (sanity check) ===")
    for name in MERGED_NAMES:
        got = int(out[f"has_{name}"].sum())
        exp = EXPECTED.get(name, "?")
        print(f"    {name:22s}: {got:6d}   (expected ~{exp})")
    print("\n[done] If these totals match the expectations, the merge is correct.")


if __name__ == "__main__":
    main()
