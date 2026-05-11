from pathlib import Path
import platform

# --- Platform-adaptive root ---
ROOT = Path(r"C:\Users\kuti\Desktop\harmonia_v2")

DATASET_DIR = ROOT / "CBIS-DDSM Dataset"
CSV_DIR = DATASET_DIR / "csv"
JPEG_DIR = DATASET_DIR / "jpeg"

OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"

MASS_TRAIN_CSV = CSV_DIR / "mass_case_description_train_set.csv"
MASS_TEST_CSV = CSV_DIR / "mass_case_description_test_set.csv"
CALC_TRAIN_CSV = CSV_DIR / "calc_case_description_train_set.csv"
CALC_TEST_CSV = CSV_DIR / "calc_case_description_test_set.csv"
DICOM_INFO_CSV = CSV_DIR / "dicom_info.csv"
META_CSV = CSV_DIR / "meta.csv"

IMAGE_SIZE = 256
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MAC_ROOT_STR = "/Users/berilcitil/Desktop/data_cleaning"
WIN_ROOT_STR = str(ROOT)

def fix_path(path_str: str) -> str:
    if not isinstance(path_str, str):
        return path_str
    path_str = path_str.replace(MAC_ROOT_STR, WIN_ROOT_STR)
    path_str = path_str.replace("/", "\\")
    if "/" in path_str:
        path_str = path_str.replace("/", "\\")
    return path_str
