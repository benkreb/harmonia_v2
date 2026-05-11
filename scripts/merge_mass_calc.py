"""
Merge Mass + Calc pipelines into combined dataset.

After running both mass and calc pipelines through step9 separately,
this script:
1. Loads step9_mass_splits.csv and step9_calc_splits.csv
2. Re-splits at patient level (70/15/15) to maintain data leakage prevention
3. Produces step9_combined_splits.csv
4. Copies all processed PNGs to a unified processed/ dir

Kullanim:
  python merge_mass_calc.py
"""
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO


UNIFIED_IMG_DIR = OUTPUT_DIR / "processed" / "images_combined"
UNIFIED_MASK_DIR = OUTPUT_DIR / "processed" / "masks_combined"


def patient_level_labels(df: pd.DataFrame) -> pd.DataFrame:
    has_malignant = df.groupby("patient_id")["label"].max().rename("patient_label")
    n_samples = df.groupby("patient_id").size().rename("n_samples_per_patient")
    return pd.concat([has_malignant, n_samples], axis=1).reset_index()


def main():
    mass = pd.read_csv(OUTPUT_DIR / "step9_mass_splits.csv")
    calc = pd.read_csv(OUTPUT_DIR / "step9_calc_splits.csv")

    print(f"[mass] {len(mass)} samples, {mass['patient_id'].nunique()} patients")
    print(f"[calc] {len(calc)} samples, {calc['patient_id'].nunique()} patients")

    # Check for patient overlap
    mass_pids = set(mass["patient_id"].unique())
    calc_pids = set(calc["patient_id"].unique())
    overlap = mass_pids & calc_pids
    if overlap:
        print(f"[warn] {len(overlap)} patients appear in both mass and calc!")
        for pid in sorted(overlap)[:5]:
            m = mass[mass["patient_id"] == pid][["pathology", "case_type"]].iloc[0]
            c = calc[calc["patient_id"] == pid][["pathology", "case_type"]].iloc[0]
            print(f"  {pid}: mass={m.to_dict()}, calc={c.to_dict()}")

    df = pd.concat([mass, calc], axis=0, ignore_index=True)
    print(f"\n[combined] {len(df)} total samples, {df['patient_id'].nunique()} patients")

    # Re-do patient-level split to maintain data leakage prevention
    pdf = patient_level_labels(df)
    print(f"\n[patient-level labels]")
    print(pdf["patient_label"].value_counts())

    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

    train_pids, temp_pids, _, _ = train_test_split(
        pdf["patient_id"], pdf["patient_label"],
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=pdf["patient_label"],
        random_state=RANDOM_SEED,
    )

    temp_labels = pdf.set_index("patient_id").loc[temp_pids, "patient_label"]
    val_pids, test_pids, _, _ = train_test_split(
        temp_pids, temp_labels,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_labels,
        random_state=RANDOM_SEED,
    )

    split_map = (
        {pid: "train" for pid in train_pids}
        | {pid: "val" for pid in val_pids}
        | {pid: "test" for pid in test_pids}
    )
    df["split"] = df["patient_id"].map(split_map)

    # Verify patient isolation
    check = df.groupby("patient_id")["split"].nunique()
    assert (check == 1).all(), "HATA: Bir hasta birden fazla split'te!"

    print("\n[sizes]")
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        print(f"  {s:5s}: {len(sub):5d} samples | "
              f"{sub['patient_id'].nunique():4d} patients | "
              f"mass={len(sub[sub['case_type']=='mass']):4d} calc={len(sub[sub['case_type']=='calc']):4d}")

    print("\n[class balance]")
    ct = pd.crosstab(df["split"], df["pathology"], margins=True, normalize=False)
    print(ct)
    print("\n[case type mix]")
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        ct2 = pd.crosstab(sub["case_type"], sub["pathology"])
        print(f"\n  {s}:")
        print(ct2)

    # Copy PNGs to unified location
    UNIFIED_IMG_DIR.mkdir(parents=True, exist_ok=True)
    UNIFIED_MASK_DIR.mkdir(parents=True, exist_ok=True)

    img_map = {}
    mask_map = {}

    for _, row in df.iterrows():
        img_src = row["processed_image_path"]
        msk_src = row["processed_mask_path"]
        name = f"{row['case_type']}_{row['patient_id']}_{row['left_or_right_breast']}_{row['image_view']}_{row['abnormality_id']}"

        img_dst = UNIFIED_IMG_DIR / f"{name}.png"
        msk_dst = UNIFIED_MASK_DIR / f"{name}.png"

        if not img_dst.exists():
            shutil.copy2(img_src, img_dst)
        if not msk_dst.exists():
            shutil.copy2(msk_src, msk_dst)

        img_map[name] = str(img_dst)
        mask_map[name] = str(msk_dst)

    df["processed_image_path"] = df.apply(
        lambda r: img_map[f"{r['case_type']}_{r['patient_id']}_{r['left_or_right_breast']}_{r['image_view']}_{r['abnormality_id']}"],
        axis=1
    )
    df["processed_mask_path"] = df.apply(
        lambda r: mask_map[f"{r['case_type']}_{r['patient_id']}_{r['left_or_right_breast']}_{r['image_view']}_{r['abnormality_id']}"],
        axis=1
    )

    out = OUTPUT_DIR / "step9_combined_splits.csv"
    df.to_csv(out, index=False)
    print(f"\n[ok] Saved: {out}")
    print(f"[ok] Unified images: {UNIFIED_IMG_DIR}")
    print(f"[ok] Unified masks:  {UNIFIED_MASK_DIR}")


if __name__ == "__main__":
    main()
