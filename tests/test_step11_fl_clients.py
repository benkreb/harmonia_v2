import pandas as pd
import pytest

from fl.partition import generate_iid_partition, generate_noniid_partition


def make_processed_metadata(n_patients: int = 40) -> pd.DataFrame:
    rows = []
    for idx in range(n_patients):
        patient_id = f"P_{idx:05d}"
        label = idx % 2
        for sample_idx in range(2):
            rows.append(
                {
                    "patient_id": patient_id,
                    "left_or_right_breast": "LEFT",
                    "image_view": "CC" if sample_idx == 0 else "MLO",
                    "abnormality_id": sample_idx + 1,
                    "pathology": "MALIGNANT" if label else "BENIGN",
                    "label": label,
                    "pathology_original": "MALIGNANT" if label else "BENIGN",
                    "breast_density": 2,
                    "assessment": 4,
                    "subtlety": 3,
                    "mask_area_ratio": 0.25,
                    "processed_image_path": f"image_{idx}_{sample_idx}.png",
                    "processed_mask_path": f"mask_{idx}_{sample_idx}.png",
                    "preprocessing": "square_pad+resize",
                    "case_type": "calc",
                }
            )
    return pd.DataFrame(rows)


def test_iid_partition_keeps_patients_disjoint_and_balanced() -> None:
    df = make_processed_metadata(400)
    partition = generate_iid_partition(df, n_clients=2, seed=42)

    clients = partition["clients"]
    p1 = set(clients["client_1"]["patient_ids"])
    p2 = set(clients["client_2"]["patient_ids"])
    assert p1.isdisjoint(p2)
    assert p1 | p2 == set(df["patient_id"].unique())
    assert sum(client["n_samples"] for client in clients.values()) == len(df)
    assert partition["stats"]["skew_metric"] < 0.10


def test_noniid_partition_has_accepted_80_20_skew() -> None:
    df = make_processed_metadata(100)
    partition = generate_noniid_partition(df, n_clients=2, seed=42)

    clients = partition["clients"]
    assert set(clients) == {"client_1", "client_2"}
    assert sum(client["n_samples"] for client in clients.values()) == len(df)
    assert 0.40 <= partition["stats"]["skew_metric"] <= 0.80


def test_partition_rejects_non_two_client_phase() -> None:
    with pytest.raises(ValueError, match="exactly 2"):
        generate_iid_partition(make_processed_metadata(), n_clients=3, seed=42)
