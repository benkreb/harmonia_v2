"""Step 11: Generate validated patient-level FL partition JSON files."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentation.config import DATA_DIR, ROOT as PROJECT_ROOT  # noqa: E402
from fl.partition import DEFAULT_SEED, generate_partitions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create FL partition JSON files.")
    parser.add_argument(
        "--scenario",
        choices=["all", "iid", "noniid", "noniid_80_20"],
        default="all",
        help="Partition scenario to generate.",
    )
    parser.add_argument("--n-clients", type=int, default=2, help="Must be 2 for this phase.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory with final .npy data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "partitions",
        help="Directory for partition JSON outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    written = generate_partitions(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        scenario=args.scenario,
        n_clients=args.n_clients,
        seed=args.seed,
    )
    for scenario, path in written.items():
        print(f"[ok] Saved {scenario} partition to {path}")


if __name__ == "__main__":
    main()
