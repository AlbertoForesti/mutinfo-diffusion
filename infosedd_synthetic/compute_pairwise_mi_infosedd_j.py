#!/usr/bin/env python

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datamodule import CSVDataModule
from mi_estimator import MIEstimator
from infosedd_utils import get_score_fn
import graph_lib
import noise_lib


def _find_latest_infosedd_j_checkpoint(checkpoints_root: Path) -> Path:
    infosedd_root = checkpoints_root / "infosedd"
    if not infosedd_root.exists():
        raise FileNotFoundError(
            f"No infosedd checkpoint directory found at {infosedd_root}."
        )

    last_candidates = list(infosedd_root.glob("**/last.ckpt"))
    if last_candidates:
        return max(last_candidates, key=lambda p: p.stat().st_mtime)

    ckpt_candidates = list(infosedd_root.glob("**/*.ckpt"))
    if ckpt_candidates:
        return max(ckpt_candidates, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(f"No checkpoint files found under {infosedd_root}.")


def _parse_col_selector(selector: str, columns: list[str], kind: str) -> int:
    if selector.isdigit():
        idx = int(selector)
        if idx < 0 or idx >= len(columns):
            raise ValueError(
                f"{kind} index {idx} is out of range [0, {len(columns) - 1}]."
            )
        return idx

    if selector not in columns:
        raise ValueError(
            f"{kind} selector '{selector}' not found. Available {kind} columns: {columns}"
        )
    return columns.index(selector)


def _estimate_pair_mi(
    model: MIEstimator,
    graph,
    noise,
    x: torch.Tensor,
    y: torch.Tensor,
    x_idx: int,
    y_idx: int,
    mc_estimates: int,
) -> tuple[float, float]:
    score_fn = get_score_fn(model.backbone, train=False, sampling=True)

    batch = torch.cat([x, y], dim=-1)
    n_total = batch.shape[1]
    y_global_idx = x.shape[1] + y_idx
    keep_indices = torch.tensor([x_idx, y_global_idx], device=batch.device)
    keep_mask = torch.zeros(n_total, dtype=torch.bool, device=batch.device)
    keep_mask[keep_indices] = True

    mi_estimates = []
    for _ in range(mc_estimates):
        t = torch.rand(batch.shape[0], 1, device=batch.device)
        sigma, dsigma = noise(t)

        perturbed = graph.sample_transition(batch, sigma)
        perturbed_joint = perturbed.clone()
        perturbed_joint[:, ~keep_mask] = graph.dim - 1

        perturbed_marginal_x = perturbed_joint.clone()
        perturbed_marginal_x[:, y_global_idx] = graph.dim - 1

        perturbed_marginal_y = perturbed_joint.clone()
        perturbed_marginal_y[:, x_idx] = graph.dim - 1

        score_joint = score_fn(perturbed_joint, sigma)[:, keep_indices]
        score_marginal_x = score_fn(perturbed_marginal_x, sigma)[:, x_idx : x_idx + 1]
        score_marginal_y = score_fn(perturbed_marginal_y, sigma)[:, y_global_idx : y_global_idx + 1]
        score_marginal = torch.cat([score_marginal_x, score_marginal_y], dim=1)

        perturbed_pair = perturbed_joint[:, keep_indices]
        divergence_estimate = graph.score_divergence(
            score_joint,
            score_marginal,
            dsigma,
            perturbed_pair,
        )
        mi_estimates.append(divergence_estimate.mean().item())

    mi_tensor = torch.tensor(mi_estimates)
    return mi_tensor.mean().item(), mi_tensor.std(unbiased=False).item()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute MI for CSV pairs (X_i, Y_j) using latest InfoSEDD-j checkpoint."
    )
    parser.add_argument("--x", default=None, help="X selector: index (e.g. 0) or exact column name")
    parser.add_argument("--y", default=None, help="Y selector: index (e.g. 0) or exact column name")
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Compute MI for all X/Y pairs.",
    )
    parser.add_argument("--mc-estimates", type=int, default=10, help="Number of Monte Carlo estimates")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path. If omitted, latest infosedd-j checkpoint is used.",
    )
    parser.add_argument(
        "--checkpoints-root",
        default="checkpoints",
        help="Checkpoint root directory used when --checkpoint is not provided.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device for MI computation.",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="List available CSV X/Y columns and exit.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output YAML path. If omitted with --all-pairs, a timestamped path in mi_statistics is used.",
    )
    parser.add_argument(
        "--config",
        default="configs/data/example_csv.yaml",
        help="Path to datamodule config YAML.",
    )
    args = parser.parse_args()

    data_cfg = OmegaConf.load(args.config).config
    datamodule = CSVDataModule(data_cfg)
    datamodule.setup()

    x_columns = list(data_cfg.x_col)
    y_columns = list(data_cfg.y_col)
    if args.list_columns:
        print("X columns:")
        for i, name in enumerate(x_columns):
            print(f"  [{i}] {name}")
        print("Y columns:")
        for i, name in enumerate(y_columns):
            print(f"  [{i}] {name}")
        return

    if not args.all_pairs and (args.x is None or args.y is None):
        raise ValueError("Either provide --x and --y, or use --all-pairs.")

    x_idx = _parse_col_selector(args.x, x_columns, "x") if args.x is not None else None
    y_idx = _parse_col_selector(args.y, y_columns, "y") if args.y is not None else None

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else _find_latest_infosedd_j_checkpoint(Path(args.checkpoints_root))
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    root_cfg = OmegaConf.load("configs/config.yaml")
    estimator_cfg = OmegaConf.load("configs/estimator/infosedd_j.yaml").config
    estimator_cfg.seq_length = data_cfg.seq_length
    estimator_cfg.alphabet_size = data_cfg.alphabet_size
    estimator_cfg.sigma_dim = root_cfg.sigma_dim
    estimator_cfg.init_dim = root_cfg.init_dim
    estimator_cfg.resnet_block_groups = root_cfg.resnet_block_groups
    graph = graph_lib.get_graph(estimator_cfg)
    noise = noise_lib.get_noise(estimator_cfg)

    estimator = MIEstimator(estimator_cfg)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    estimator.load_state_dict(ckpt["state_dict"], strict=True)
    estimator.eval()
    estimator.to(args.device)

    x = datamodule.data.tensors[0].to(args.device)
    y = datamodule.data.tensors[1].to(args.device)
    metadata = {
        "created_at": datetime.now().isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "device": args.device,
        "mc_estimates": args.mc_estimates,
        "dataset_path": str(data_cfg.file_path),
        "n_samples": int(x.shape[0]),
        "x_dim": int(x.shape[1]),
        "y_dim": int(y.shape[1]),
        "alphabet_size": int(data_cfg.alphabet_size),
        "float_handling": str(data_cfg.float_handling),
        "float_bins": int(data_cfg.float_bins),
        "estimator": "infosedd_j",
        "variant": "j",
    }

    if args.all_pairs:
        pair_results = []
        for xi in range(len(x_columns)):
            for yj in range(len(y_columns)):
                mi_mean, mi_std = _estimate_pair_mi(
                    model=estimator,
                    graph=graph,
                    noise=noise,
                    x=x,
                    y=y,
                    x_idx=xi,
                    y_idx=yj,
                    mc_estimates=args.mc_estimates,
                )
                pair_results.append(
                    {
                        "x_index": xi,
                        "x_name": x_columns[xi],
                        "y_index": yj,
                        "y_name": y_columns[yj],
                        "mi_mean": float(mi_mean),
                        "mi_std": float(mi_std),
                    }
                )
                print(
                    f"x[{xi}]={x_columns[xi]} y[{yj}]={y_columns[yj]} "
                    f"mi_mean={mi_mean:.8f} mi_std={mi_std:.8f}"
                )

        output = {
            "metadata": metadata,
            "x_columns": x_columns,
            "y_columns": y_columns,
            "pair_results": pair_results,
        }

        if args.output_path is None:
            output_path = Path(
                f"mi_statistics/infosedd_j_pairs/seq_length={data_cfg.seq_length}/"
                f"seed={root_cfg.seed}/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.yaml"
            )
        else:
            output_path = Path(args.output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(output, f, sort_keys=False)
        print(f"Saved pair MI results to {output_path}")
    else:
        mi_mean, mi_std = _estimate_pair_mi(
            model=estimator,
            graph=graph,
            noise=noise,
            x=x,
            y=y,
            x_idx=x_idx,
            y_idx=y_idx,
            mc_estimates=args.mc_estimates,
        )
        print(f"checkpoint={checkpoint_path}")
        print(f"x[{x_idx}]={x_columns[x_idx]} y[{y_idx}]={y_columns[y_idx]}")
        print(f"mi_mean={mi_mean:.8f} mi_std={mi_std:.8f} mc_estimates={args.mc_estimates}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
