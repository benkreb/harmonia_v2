"""
Step 3: Siniflandirma icin etiket duzenlemesi.
Kullanim: python step3_consolidate_labels.py [--calc]
"""
import sys
import pandas as pd
from config import OUTPUT_DIR


LABEL_MAP = {
    "BENIGN": "BENIGN",
    "BENIGN_WITHOUT_CALLBACK": "BENIGN",
    "MALIGNANT": "MALIGNANT",
}
BINARY_MAP = {"BENIGN": 0, "MALIGNANT": 1}


def main():
    case_type = "calc" if "--calc" in sys.argv else "mass"
    in_path = OUTPUT_DIR / f"step2_{case_type}_joined.csv"
    df = pd.read_csv(in_path)
    print(f"[in] {in_path.name} shape: {df.shape}")

    print("\n=== Once (orijinal) pathology dagilimi ===")
    print(df["pathology"].value_counts(dropna=False))

    # Orijinal etiketi sakla
    df["pathology_original"] = df["pathology"]

    # 2-sinifli konsolidasyon
    df["pathology"] = df["pathology"].map(LABEL_MAP)

    # Map sonrasi null kalirsa bilinmeyen etiket demektir
    unknown = df["pathology"].isna().sum()
    if unknown:
        print(f"[warn] {unknown} satirda bilinmeyen etiket (map disinda)")

    # Binary hedef sutunu (model icin kullanisli)
    df["label"] = df["pathology"].map(BINARY_MAP)

    print("\n=== Sonra (2-sinifli) pathology dagilimi ===")
    print(df["pathology"].value_counts(dropna=False))

    print("\n=== pathology_original -> pathology eslemesi ===")
    print(pd.crosstab(df["pathology_original"], df["pathology"], dropna=False))

    print("\n=== Sinif orani ===")
    total = len(df)
    for cls, n in df["pathology"].value_counts().items():
        print(f"  {cls}: {n} ({n/total:.1%})")

    out = OUTPUT_DIR / f"step3_{case_type}_labels_consolidated.csv"
    df.to_csv(out, index=False)
    print(f"\n[ok] Kaydedildi: {out}")


if __name__ == "__main__":
    main()
