"""
Step 1: CBIS-DDSM CSV'lerini yukle, sutun isimlerini normalize et,
train/test'i tek bir DataFrame'de birlestir, ilk kesif (EDA).

Kullanim:
  python step1_load_and_merge.py              # varsayilan: mass
  python step1_load_and_merge.py --calc       # calc CSVs
"""
import sys
import pandas as pd
from config import MASS_TRAIN_CSV, MASS_TEST_CSV, CALC_TRAIN_CSV, CALC_TEST_CSV, OUTPUT_DIR

CASE_SPECIFIC_COLS = {
    "mass": {"mass_shape", "mass_margins"},
    "calc": {"calc_type", "calc_distribution"},
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def load_data(train_csv, test_csv, case_type: str) -> pd.DataFrame:
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    print(f"[raw] {case_type}_train shape: {train.shape}")
    print(f"[raw] {case_type}_test  shape: {test.shape}")
    print(f"[raw] {case_type}_train columns: {list(train.columns)}")

    train = normalize_columns(train)
    test = normalize_columns(test)

    train["source_split"] = "train"
    test["source_split"] = "test"

    df = pd.concat([train, test], axis=0, ignore_index=True)
    df["case_type"] = case_type
    print(f"[merged] combined shape: {df.shape}")
    return df


def eda(df: pd.DataFrame, case_type: str) -> None:
    print("\n=== Normalize edilmis sutunlar ===")
    for c in df.columns:
        print(f"  - {c}")

    print("\n=== Dtypes ===")
    print(df.dtypes)

    print("\n=== Null sayilari (sifirdan buyuk olanlar) ===")
    nulls = df.isna().sum()
    print(nulls[nulls > 0].sort_values(ascending=False))

    extra_cols = CASE_SPECIFIC_COLS.get(case_type, set())
    base_cols = ["pathology", "assessment", "subtlety", "breast_density",
                 "left_or_right_breast", "image_view"]
    print("\n=== Unique degerler (kritik kategorik sutunlar) ===")
    for col in base_cols + sorted(extra_cols):
        if col in df.columns:
            vals = df[col].value_counts(dropna=False)
            print(f"\n[{col}] ({df[col].nunique(dropna=False)} unique)")
            print(vals.head(15))

    print("\n=== Hasta sayisi ===")
    if "patient_id" in df.columns:
        print(f"  Toplam satir: {len(df)}")
        print(f"  Unique patient_id: {df['patient_id'].nunique()}")
        print(f"  Unique (patient_id, left_or_right_breast, image_view): "
              f"{df.groupby(['patient_id','left_or_right_breast','image_view']).ngroups}")

    print("\n=== Pathology dagilimi ===")
    print(df["pathology"].value_counts(dropna=False))


def main():
    case_type = "calc" if "--calc" in sys.argv else "mass"
    train_csv = CALC_TRAIN_CSV if case_type == "calc" else MASS_TRAIN_CSV
    test_csv = CALC_TEST_CSV if case_type == "calc" else MASS_TEST_CSV

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(train_csv, test_csv, case_type)
    eda(df, case_type)

    out_path = OUTPUT_DIR / f"step1_{case_type}_merged.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[ok] Kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
