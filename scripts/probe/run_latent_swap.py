import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging

from scripts.probe.collect_probe_features import (
    build_label_maps,
    build_sample_indices_for_labeled_episodes,
    extract_episode_id,
    infer_episode_span_from_dataset,
    infer_label_keys,
    load_task_name_map,
    map_variant_label,
    normalize_obs_to_model_input,
    pick_label,
    resolve_label_variant_spec,
)

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Offline video/world latent cache swap for FastWAM IDM probes.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--suite", required=True, help="Kept for CLI compatibility with probe scripts.")
    p.add_argument("--tasks", required=True, help="Comma-separated task ids, e.g. 0,1,2")
    p.add_argument("--label_config", required=True)
    p.add_argument("--label_variant", choices=["native", "static_active", "open_close"], default="open_close")
    p.add_argument("--target", default=None, help="Label key used to choose A/B pairs. Defaults to variant/native first key.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_episodes", type=int, default=1)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max_pairs", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_inference_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe_path", default=None, help="Optional trained probe checkpoint/state dict for action_out transfer metrics.")
    p.add_argument("--probe_representation", default="action_out")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _to_container(path: str):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _load_cfg_with_hydra(config_path: str):
    cfg_path = Path(config_path).resolve()
    configs_dir = cfg_path.parent.parent if cfg_path.parent.name == "task" else cfg_path.parent
    task_name = cfg_path.stem
    with initialize_config_dir(version_base=None, config_dir=str(configs_dir)):
        return compose(config_name="train", overrides=[f"task={task_name}"])


def _tensor_to_list(x: torch.Tensor | None):
    if x is None:
        return None
    return x.detach().float().cpu().tolist()


def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a.detach().float().cpu() - b.detach().float().cpu()).item())


def _gripper_value(action: torch.Tensor) -> float:
    # Convention in common LIBERO action vectors: final dim is gripper command.
    if action.ndim == 2:
        return float(action[-1, -1].detach().float().cpu().item())
    return float(action.reshape(-1)[-1].detach().float().cpu().item())


def _move_cache_to_cpu(cache: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_seq_len": int(cache["video_seq_len"]),
        "video_tokens_per_frame": int(cache.get("video_tokens_per_frame", 0)),
        "kv_cache": [
            {"k": layer["k"].detach().cpu(), "v": layer["v"].detach().cpu()}
            for layer in cache["kv_cache"]
        ],
        "video_out": None if cache.get("video_out") is None else cache["video_out"].detach().cpu(),
    }


def _move_cache_to_device(cache: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "video_seq_len": int(cache["video_seq_len"]),
        "video_tokens_per_frame": int(cache.get("video_tokens_per_frame", 0)),
        "kv_cache": [
            {"k": layer["k"].to(device=device, dtype=dtype), "v": layer["v"].to(device=device, dtype=dtype)}
            for layer in cache["kv_cache"]
        ],
        "video_out": None if cache.get("video_out") is None else cache["video_out"].to(device=device, dtype=dtype),
    }


def load_probe(probe_path: str | None):
    if not probe_path:
        return None
    obj = torch.load(probe_path, map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], torch.nn.Module):
        obj["model"].eval()
        return obj["model"]
    logger.warning("Unsupported --probe_path format; action_out probe transfer metrics will be skipped: %s", probe_path)
    return None


def probe_predict(probe, feat: torch.Tensor | None) -> int | None:
    if probe is None or feat is None:
        return None
    with torch.no_grad():
        x = feat.detach().float().cpu()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        logits = probe(x)
        return int(logits.argmax(dim=1).item())


def make_labeled_records(dataset: Any, label_cfg: dict[str, Any], variant_spec: dict[str, Any] | None, label_keys: list[str], label_maps: dict[str, dict[str, int]], stride: int, dry_run: bool) -> list[dict[str, Any]]:
    episode_blocks = label_cfg.get("episodes", {}) or label_cfg.get("tasks", {})
    labeled_episode_ids = {int(k) for k in episode_blocks.keys()}
    sample_indices = build_sample_indices_for_labeled_episodes(dataset, labeled_episode_ids, max(1, stride))
    if dry_run:
        sample_indices = sample_indices[:16]

    records: list[dict[str, Any]] = []
    for idx in sample_indices:
        sample = dataset[idx]
        episode_span = infer_episode_span_from_dataset(dataset, idx)
        sample_episode_id = extract_episode_id(sample)
        if episode_span is None:
            if sample_episode_id is None:
                continue
            episode_key = int(sample_episode_id)
            episode_len = int(sample.get("episode_len", sample.get("length", sample["video"].shape[0])))
            timestep = int(sample.get("timestep", min(idx, episode_len - 1)))
        else:
            episode_key, episode_start, episode_end = episode_span
            episode_len = max(1, int(episode_end) - int(episode_start))
            timestep = int(idx) - int(episode_start)
        if episode_key not in episode_blocks:
            continue
        span = pick_label(episode_blocks[episode_key], timestep, episode_len)
        if variant_spec is None:
            label_names = {k: span[k] for k in label_keys}
        else:
            raw_label = span[variant_spec["source_key"]]
            mapped = map_variant_label(raw_label, variant_spec)
            if mapped is None:
                continue
            label_names = {variant_spec["output_key"]: mapped, f"{variant_spec['source_key']}_raw": raw_label}
        labels = {k: int(label_maps[k][label_names[k]]) for k in label_keys}
        records.append({
            "idx": int(idx),
            "episode_id": int(episode_key),
            "timestep": int(timestep),
            "episode_len": int(episode_len),
            "label_names": label_names,
            "labels": labels,
        })
    return records


def make_pairs(records: list[dict[str, Any]], target: str, max_pairs: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_label: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        by_label.setdefault(int(rec["labels"][target]), []).append(rec)
    labels = sorted(by_label)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for a_label in labels:
        for b_label in labels:
            if a_label == b_label:
                continue
            for a, b in zip(by_label[a_label], by_label[b_label]):
                pairs.append((a, b))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def infer_one(model, sample, processor: FastWAMProcessor, args, load_text_encoder: bool, external_video_cache=None):
    model._reset_probe_cache()
    x, proprio, prompt, context, context_mask = normalize_obs_to_model_input(sample, processor, args.device, model.torch_dtype)
    kwargs = {
        "input_image": x,
        "action_horizon": int(sample.get("action_horizon", 1)),
        "num_video_frames": int(sample.get("num_frames", 33)),
        "proprio": proprio,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "return_video_cache": True,
        "external_video_cache": external_video_cache,
    }
    if load_text_encoder:
        kwargs["prompt"] = prompt
    else:
        if context is None or context_mask is None:
            raise ValueError("Sample is missing precomputed context/context_mask and text encoder is disabled.")
        if isinstance(context, torch.Tensor) and context.ndim == 2:
            context = context.unsqueeze(0)
        if isinstance(context_mask, torch.Tensor) and context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        kwargs.update({"prompt": None, "context": context, "context_mask": context_mask})
    out = model.infer_action(**kwargs)
    feats = model.get_probe_features()
    return {
        "action": out["action"].detach().cpu(),
        "video_cache": _move_cache_to_cpu(out["video_cache"]),
        "action_out": None if "action_out" not in feats else feats["action_out"].detach().cpu(),
        "features": {k: v.detach().cpu() for k, v in feats.items()},
        "prompt": prompt,
    }


def main():
    register_default_resolvers()
    args = parse_args()
    setup_logging(log_level=20)
    del args.suite

    cfg = _load_cfg_with_hydra(args.config)
    label_cfg = _to_container(args.label_config)
    variant_spec = resolve_label_variant_spec(args.label_variant, label_cfg)
    if variant_spec is None:
        label_keys = infer_label_keys(label_cfg)
        label_maps, label_names = build_label_maps(label_cfg)
    else:
        variant_spec["name"] = args.label_variant
        label_keys = [variant_spec["output_key"]]
        label_maps = variant_spec["label_maps"]
        label_names = variant_spec["label_names"]
    target = args.target or label_keys[0]
    if target not in label_keys:
        raise KeyError(f"Target {target!r} is not available; label keys are {label_keys}")

    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=args.device)
    if model.__class__.__name__ != "FastWAMIDM":
        raise TypeError(f"Latent swap requires FastWAMIDM, got {type(model)!r}")
    model.load_checkpoint(args.checkpoint)
    model.eval()
    model.enable_probe = True
    load_text_encoder = bool(OmegaConf.select(cfg, "model.load_text_encoder", default=False))

    task_name_map = load_task_name_map(args.data_root)
    requested_task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]
    requested_task_names = [task_name_map[t] for t in requested_task_ids if t in task_name_map]
    dataset_cfg = OmegaConf.to_container(cfg.data.train, resolve=True)
    dataset_cfg["dataset_dirs"] = [args.data_root]
    dataset_cfg["selected_tasks"] = requested_task_names if requested_task_names else None
    if args.max_episodes > 0:
        dataset_cfg["max_episodes_per_task"] = args.max_episodes
    else:
        dataset_cfg.pop("max_episodes_per_task", None)
    dataset = instantiate(OmegaConf.create(dataset_cfg))

    stats_path = cfg.data.train.get("pretrained_norm_stats", None)
    if stats_path is None:
        ckpt_stats = Path(args.checkpoint).resolve().parent / "dataset_stats.json"
        data_root_stats = Path(args.data_root) / "dataset_stats.json"
        stats_path = str(ckpt_stats if ckpt_stats.exists() else data_root_stats)
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(stats_path))

    records = make_labeled_records(dataset, label_cfg, variant_spec, label_keys, label_maps, args.stride, args.dry_run)
    pairs = make_pairs(records, target, max(1, args.max_pairs))
    if not pairs:
        raise RuntimeError(f"No cross-label A/B pairs found for target={target}")
    probe = load_probe(args.probe_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    examples = []
    for pair_id, (a_rec, b_rec) in enumerate(pairs):
        sample_a = dataset[a_rec["idx"]]
        sample_b = dataset[b_rec["idx"]]
        normal_a = infer_one(model, sample_a, processor, args, load_text_encoder)
        normal_b = infer_one(model, sample_b, processor, args, load_text_encoder)
        b_cache_on_device = _move_cache_to_device(normal_b["video_cache"], model.device, model.torch_dtype)
        swapped = infer_one(model, sample_a, processor, args, load_text_encoder, external_video_cache=b_cache_on_device)

        dist_a_to_b = _l2(normal_a["action"], normal_b["action"])
        dist_swap_to_b = _l2(swapped["action"], normal_b["action"])
        dist_swap_to_a = _l2(swapped["action"], normal_a["action"])
        grip_a = _gripper_value(normal_a["action"])
        grip_b = _gripper_value(normal_b["action"])
        grip_s = _gripper_value(swapped["action"])
        grip_shift_success = abs(grip_s - grip_b) < abs(grip_a - grip_b)
        action_shift_success = dist_swap_to_b < dist_a_to_b
        pred_a = probe_predict(probe, normal_a["action_out"])
        pred_b = probe_predict(probe, normal_b["action_out"])
        pred_s = probe_predict(probe, swapped["action_out"])
        probe_transfer_success = None if pred_s is None else (pred_s == int(b_rec["labels"][target]) and pred_s != int(a_rec["labels"][target]))

        row = {
            "pair_id": pair_id,
            "a_idx": a_rec["idx"],
            "b_idx": b_rec["idx"],
            "a_episode_id": a_rec["episode_id"],
            "b_episode_id": b_rec["episode_id"],
            "a_timestep": a_rec["timestep"],
            "b_timestep": b_rec["timestep"],
            "target": target,
            "a_label": a_rec["label_names"][target],
            "b_label": b_rec["label_names"][target],
            "a_label_id": a_rec["labels"][target],
            "b_label_id": b_rec["labels"][target],
            "dist_a_to_b": dist_a_to_b,
            "dist_swap_to_b": dist_swap_to_b,
            "dist_swap_to_a": dist_swap_to_a,
            "action_distance_delta_to_b": dist_a_to_b - dist_swap_to_b,
            "action_shift_success": action_shift_success,
            "gripper_a": grip_a,
            "gripper_b": grip_b,
            "gripper_swap": grip_s,
            "gripper_delta_to_b": abs(grip_a - grip_b) - abs(grip_s - grip_b),
            "gripper_shift_success": grip_shift_success,
            "probe_pred_a": pred_a,
            "probe_pred_b": pred_b,
            "probe_pred_swap": pred_s,
            "probe_transfer_success": probe_transfer_success,
        }
        rows.append(row)
        examples.append({
            "metadata": row,
            "a_normal": {"action": _tensor_to_list(normal_a["action"]), "action_out": _tensor_to_list(normal_a["action_out"])},
            "b_normal": {"action": _tensor_to_list(normal_b["action"]), "action_out": _tensor_to_list(normal_b["action_out"])},
            "a_with_b_cache": {"action": _tensor_to_list(swapped["action"]), "action_out": _tensor_to_list(swapped["action_out"])},
        })
        logger.info("pair %d: A=%s B=%s delta_to_b=%.4f gripper_delta=%.4f", pair_id, row["a_label"], row["b_label"], row["action_distance_delta_to_b"], row["gripper_delta_to_b"])

    summary = {
        "num_pairs": len(rows),
        "target": target,
        "label_names": label_names,
        "action_shift_success_rate": float(np.mean([r["action_shift_success"] for r in rows])),
        "gripper_shift_success_rate": float(np.mean([r["gripper_shift_success"] for r in rows])),
        "mean_action_distance_delta_to_b": float(np.mean([r["action_distance_delta_to_b"] for r in rows])),
        "mean_gripper_delta_to_b": float(np.mean([r["gripper_delta_to_b"] for r in rows])),
    }
    probe_vals = [r["probe_transfer_success"] for r in rows if r["probe_transfer_success"] is not None]
    if probe_vals:
        summary["probe_transfer_success_rate"] = float(np.mean(probe_vals))

    csv_path = out_dir / "latent_swap_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "latent_swap_results.json", "w") as f:
        json.dump({"summary": summary, "rows": rows, "examples": examples}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(csv_path)


if __name__ == "__main__":
    main()
