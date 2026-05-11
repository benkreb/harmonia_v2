"""Verify that saved PHE parameters evaluate consistently after decryption."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fl.config import FLBenchmarkConfig
from fl.factories import build_loss_from_config, build_model_from_config
from fl.parameters import deserialize_parameters_from_file, load_parameters_into_model
from fl.simulation import build_global_loader
from segmentation.evaluation.fl_evaluator import run_segmentation_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("benchmark/configs/f2_iid_phe.yaml"))
    args = parser.parse_args()

    config = FLBenchmarkConfig.from_yaml(args.config)
    if args.params.parent.name != config.name:
        config.name = args.params.parent.name
        config.output_dir = args.params.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config.segmentation).to(device)
    loss_fn = build_loss_from_config(config.segmentation)
    load_parameters_into_model(model, deserialize_parameters_from_file(args.params))
    val_loader = build_global_loader("val", config)
    loss, metrics = run_segmentation_eval(model, val_loader, device, loss_fn, use_amp=config.client.use_amp)
    print(f"val_loss={loss:.6f} val_dice={metrics['dice']:.6f}")


if __name__ == "__main__":
    main()

