#!/usr/bin/env python3
"""Command-line entry point for Harmonia FL benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.aggregate_results import generate_report
from fl.config import FLBenchmarkConfig
from fl.partition import generate_partitions
from segmentation.config import DATA_DIR, ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmonia Federated Learning Benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    part = subparsers.add_parser("partition", help="Generate FL partition JSON files")
    part.add_argument("--scenario", choices=["all", "iid", "noniid", "noniid_80_20"], default="all")
    part.add_argument("--n-clients", type=int, default=2)
    part.add_argument("--seed", type=int, default=42)
    part.add_argument("--data-dir", type=Path, default=DATA_DIR)
    part.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "partitions")

    sim = subparsers.add_parser("simulate", help="Run one FL benchmark config")
    sim.add_argument("--config", type=Path, required=True)
    sim.add_argument("--smoke", action="store_true")
    sim.add_argument("--resume", action="store_true", help="Reserved for PRD compatibility")

    bench = subparsers.add_parser("benchmark", help="Run all F-configs")
    bench.add_argument("--all", action="store_true")
    bench.add_argument("--no-smoke", action="store_true")

    subparsers.add_parser("report", help="Aggregate benchmark report")

    args = parser.parse_args()
    if args.command == "partition":
        written = generate_partitions(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            scenario=args.scenario,
            n_clients=args.n_clients,
            seed=args.seed,
        )
        for scenario, path in written.items():
            print(f"[ok] {scenario}: {path}")
    elif args.command == "simulate":
        from fl.simulation import run_simulation

        if args.resume:
            print("[warn] --resume is accepted but full resume is not implemented; starting from config.")
        config = FLBenchmarkConfig.from_yaml(args.config)
        if args.smoke:
            config.apply_smoke_overrides()
        run_simulation(config)
    elif args.command == "benchmark":
        from benchmark.run_benchmark import run_all

        if not args.all:
            raise SystemExit("Use --all to run all benchmark configs.")
        run_all(smoke_first=not args.no_smoke)
    elif args.command == "report":
        generate_report()


if __name__ == "__main__":
    main()

