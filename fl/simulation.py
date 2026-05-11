"""Simulation entry point for Harmonia FL benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform

import flwr as fl
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters
from torch.utils.data import DataLoader

from fl.client import HarmoniaFLClient
from fl.communication import CommunicationTracker
from fl.config import FLBenchmarkConfig
from fl.factories import build_loss_from_config, build_model_from_config
from fl.parameters import get_parameters_as_numpy, serialize_parameters_to_file
from fl.seeding import set_global_seeds
from fl.server import HarmoniaFedAvg, make_server_eval_fn, weighted_mean_metrics
from segmentation.data import CBISDDSMDataset, FLPartitionedDataset, get_train_transforms, get_val_transforms
from segmentation.data.transforms import ComposeTransforms
from segmentation.evaluation.fl_evaluator import evaluate_on_test
from segmentation.training.fl_trainer import FLLocalTrainer


@dataclass
class SimulationResult:
    history: object
    final_test: dict


def run_simulation(config: FLBenchmarkConfig) -> SimulationResult:
    set_global_seeds(config.seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config_snapshot.yaml")

    device = _resolve_device(config.segmentation.training.device)
    model = build_model_from_config(config.segmentation).to(device)
    loss_fn = build_loss_from_config(config.segmentation)
    initial_params = get_parameters_as_numpy(model)
    serialize_parameters_to_file(initial_params, output_dir / "initial_params.pt")

    val_loader = build_global_loader("val", config)
    test_loader = build_global_loader("test", config)
    communication_tracker = CommunicationTracker(output_dir / "communication_log.csv")
    crypto = _build_crypto(config)
    strategy_ref: dict = {}
    strategy = HarmoniaFedAvg(
        output_dir=output_dir,
        fl_config=config,
        communication_tracker=communication_tracker,
        crypto=crypto,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=config.server.min_fit_clients,
        min_available_clients=config.server.min_available_clients,
        initial_parameters=ndarrays_to_parameters(
            crypto.encrypt_parameters(initial_params) if crypto else initial_params
        ),
        evaluate_fn=make_server_eval_fn(
            model=model,
            val_loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            output_dir=output_dir,
            strategy_ref=strategy_ref,
            use_amp=config.client.use_amp,
        ),
        on_fit_config_fn=lambda r: {
            "local_epochs": config.client.local_epochs,
            "round_num": r,
        },
        fit_metrics_aggregation_fn=weighted_mean_metrics,
    )
    strategy_ref["strategy"] = strategy

    def client_fn(cid: str):
        client_idx = int(cid) + 1 if cid.isdigit() else int(cid.split("_")[-1])
        client_id = f"client_{client_idx}"
        return build_fl_client(client_id, config, crypto=crypto).to_client()

    if platform.system() == "Windows":
        print("[warn] Windows Ray startup is unreliable here; using sequential local FedAvg.")
        history = _run_sequential_fedavg(config, strategy, initial_params, val_loader, device, loss_fn)
    else:
        try:
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=config.partition.n_clients,
                config=fl.server.ServerConfig(num_rounds=config.server.max_rounds),
                strategy=strategy,
                client_resources={
                    "num_cpus": config.simulation.num_cpus_per_client,
                    "num_gpus": config.simulation.num_gpus_per_client,
                },
                ray_init_args={"log_to_driver": config.simulation.ray_log_to_driver},
            )
        except OSError as exc:
            print(f"[warn] Ray simulation failed ({exc}); falling back to sequential local FedAvg.")
            history = _run_sequential_fedavg(config, strategy, initial_params, val_loader, device, loss_fn)

    final_test = evaluate_on_test(output_dir / "best_params.pt", test_loader, config)
    return SimulationResult(history=history, final_test=final_test)


def _run_sequential_fedavg(
    config: FLBenchmarkConfig,
    strategy: HarmoniaFedAvg,
    initial_params: list[np.ndarray],
    val_loader: DataLoader,
    device: torch.device,
    loss_fn,
) -> dict:
    clients = [
        build_fl_client(f"client_{idx + 1}", config)
        for idx in range(config.partition.n_clients)
    ]
    server_model = build_model_from_config(config.segmentation).to(device)
    params = initial_params
    history = {"mode": "sequential_fallback", "rounds": []}

    for round_num in range(1, config.server.max_rounds + 1):
        strategy._round_starts[round_num] = __import__("time").time()
        client_results = []
        for client in clients:
            strategy.communication_tracker.record_download(
                round_num, client._client_id, sum(p.nbytes for p in params), encrypted=False
            )
            new_params, n_samples, metrics = client.fit(
                [param.copy() for param in params],
                {"local_epochs": config.client.local_epochs, "round_num": round_num},
            )
            strategy.communication_tracker.record_upload(
                round_num, client._client_id, sum(p.nbytes for p in new_params), encrypted=False
            )
            client_results.append((new_params, n_samples, metrics))

        total = sum(n for _, n, _ in client_results)
        weights = [n / total for _, n, _ in client_results]
        params = [
            np.asarray(sum(
                weight * np.asarray(client_params[param_idx])
                for weight, (client_params, _, _) in zip(weights, client_results)
            ), dtype=np.float32)
            for param_idx in range(len(params))
        ]
        strategy._last_fit_metrics[round_num] = {
            key: sum(weight * float(metrics.get(key, 0.0)) for weight, (_, _, metrics) in zip(weights, client_results))
            for key in ["train_loss", "train_dice", "train_iou"]
        }

        from fl.parameters import load_parameters_into_model
        from segmentation.evaluation.fl_evaluator import run_segmentation_eval

        load_parameters_into_model(server_model, params)
        loss, metrics = run_segmentation_eval(
            server_model, val_loader, device, loss_fn, use_amp=config.client.use_amp
        )
        strategy.record_server_eval(round_num, loss, metrics, params)
        history["rounds"].append({"round": round_num, "val_dice": metrics.get("dice", 0.0)})
        if strategy.should_stop:
            break
    return history


def build_fl_client(client_id: str, config: FLBenchmarkConfig, crypto=None) -> HarmoniaFLClient:
    device = _resolve_device(config.segmentation.training.device)
    model = build_model_from_config(config.segmentation)
    loss_fn = build_loss_from_config(config.segmentation)
    trainer = FLLocalTrainer(model, loss_fn, config.client, device)
    dataset = FLPartitionedDataset(
        data_dir=config.segmentation.data.data_dir,
        split="train",
        partition_json_path=config.partition.partition_json_path,
        client_id=client_id,
        transform=ComposeTransforms(get_train_transforms()),
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config.client.batch_size,
        shuffle=True,
        num_workers=config.client.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    return HarmoniaFLClient(client_id, trainer, train_loader, crypto=crypto)


def build_global_loader(split: str, config: FLBenchmarkConfig) -> DataLoader:
    dataset = CBISDDSMDataset(
        data_dir=config.segmentation.data.data_dir,
        split=split,  # type: ignore[arg-type]
        transform=ComposeTransforms(get_val_transforms()),
    )
    return DataLoader(
        dataset,
        batch_size=config.server.eval_batch_size,
        shuffle=False,
        num_workers=config.client.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _build_crypto(config: FLBenchmarkConfig):
    if not config.phe.enabled:
        return None
    from fl.crypto import CryptoContext

    return CryptoContext.create_server_context(
        poly_modulus_degree=config.phe.poly_modulus_degree,
        coeff_mod_bit_sizes=config.phe.coeff_mod_bit_sizes,
        global_scale_bits=config.phe.global_scale_bits,
        chunk_size=config.phe.chunk_size,
    )

