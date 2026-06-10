import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_episodes", type=int, default=1)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--label_config", required=True)
    p.add_argument(
        "--label_variant",
        choices=["native", "static_active", "open_close"],
        default="native",
        help=(
            "Label grouping for collection. "
            "native keeps source labels (e.g. open/close/static); "
            "static_active merges open+close into active; "
            "open_close keeps only open/close samples and drops static."
        ),
    )
    p.add_argument("--inference_mode", choices=["idm"], default="idm")
    p.add_argument(
        "--representations",
        default="video_pre,video_out,action_pre,action_out,context_pure,context_with_proprio,proprio_embed",
    )
    p.add_argument("--print_feature_shapes", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _to_container(path: str):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


_LABEL_META_KEYS = frozenset({"start_ratio", "end_ratio", "start_frame", "end_frame"})


def infer_label_keys(label_cfg: dict[str, Any]) -> list[str]:
    label_vocab = label_cfg.get("label_vocab", {}) or {}
    if label_vocab:
        return [str(k) for k in label_vocab.keys()]

    episode_blocks = label_cfg.get("episodes", {}) or label_cfg.get("tasks", {})
    keys: list[str] = []
    for block in episode_blocks.values():
        for item in block.get("labels", []):
            for key in item:
                if key not in _LABEL_META_KEYS and key not in keys:
                    keys.append(key)
    if keys:
        return keys
    return ["effect", "stage", "target"]


LABEL_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "static_active": {
        "output_key": "gripper_state",
        "remap": {"static": "static", "open": "active", "close": "active"},
        "drop": set(),
        "vocab": ["static", "active"],
    },
    "open_close": {
        "output_key": "gripper_action",
        "remap": {"open": "open", "close": "close"},
        "drop": {"static"},
        "vocab": ["open", "close"],
    },
}


def resolve_label_variant_spec(variant: str, label_cfg: dict[str, Any]) -> dict[str, Any] | None:
    if variant == "native":
        return None

    spec = dict(LABEL_VARIANT_SPECS[variant])
    source_keys = infer_label_keys(label_cfg)
    if "gripper_action" in source_keys:
        spec["source_key"] = "gripper_action"
    elif len(source_keys) == 1:
        spec["source_key"] = source_keys[0]
    else:
        raise ValueError(
            f"--label_variant {variant} expects a single source label key such as gripper_action, got {source_keys}"
        )

    output_key = spec["output_key"]
    spec["label_names"] = {output_key: list(spec["vocab"])}
    spec["label_maps"] = {output_key: {name: idx for idx, name in enumerate(spec["vocab"])}}
    return spec


def map_variant_label(raw_label: str, variant_spec: dict[str, Any]) -> str | None:
    if raw_label in variant_spec["drop"]:
        return None
    mapped = variant_spec["remap"].get(raw_label)
    if mapped is None:
        raise KeyError(f"Label {raw_label!r} has no mapping for --label_variant {variant_spec['name']}")
    return mapped


def build_label_maps(label_cfg: dict[str, Any]):
    episode_blocks = label_cfg.get("episodes", {})
    if not episode_blocks:
        episode_blocks = label_cfg.get("tasks", {})

    label_vocab = label_cfg.get("label_vocab", {}) or {}
    if label_vocab:
        names = {str(k): [str(v) for v in values] for k, values in label_vocab.items()}
    else:
        names = {k: [] for k in infer_label_keys(label_cfg)}

    for block in episode_blocks.values():
        for item in block.get("labels", []):
            for key in names:
                if key in item and item[key] not in names[key]:
                    names[key].append(item[key])
    return {k: {n: i for i, n in enumerate(v)} for k, v in names.items()}, names


def pick_label(block_cfg: dict[str, Any], timestep: int, episode_len: int) -> dict[str, str]:
    ratio = 0.0 if episode_len <= 1 else float(timestep) / float(max(episode_len - 1, 1))
    for item in block_cfg.get("labels", []):
        if float(item["start_ratio"]) <= ratio < float(item["end_ratio"]):
            return item
    if block_cfg.get("labels"):
        return block_cfg["labels"][-1]
    raise KeyError("episode has no labels")


def extract_episode_id(sample: dict[str, Any]) -> int | None:
    for k in ("episode_id", "episode_index", "episode", "idx"):
        if k in sample:
            return int(sample[k])
    return None


def _as_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return [int(v) for v in values.cpu().tolist()]
    if hasattr(values, "tolist"):
        return [int(v) for v in values.tolist()]
    return [int(v) for v in values]


def get_episode_spans_from_dataset(dataset: Any) -> list[tuple[int, int, int]]:
    """Return `(real_episode_id, episode_start, episode_end)` spans for the instantiated dataset.

    `episode_data_index` is local to the selected dataset order, while annotations use
    real `episode_index`. For filtered/non-contiguous episode selections, map local
    span positions back to `lerobot_dataset.episodes` when available.
    """
    base_dataset = getattr(dataset, "lerobot_dataset", None)
    episode_data_index = getattr(base_dataset, "episode_data_index", None)
    if episode_data_index is None:
        return []

    starts = _as_int_list(episode_data_index.get("from"))
    ends = _as_int_list(episode_data_index.get("to"))
    selected_episodes = getattr(base_dataset, "episodes", None)
    if selected_episodes is None:
        real_episode_ids = list(range(len(starts)))
    else:
        real_episode_ids = [int(e) for e in selected_episodes]

    spans: list[tuple[int, int, int]] = []
    for local_ep_idx, (start, end) in enumerate(zip(starts, ends)):
        real_ep_id = real_episode_ids[local_ep_idx] if local_ep_idx < len(real_episode_ids) else local_ep_idx
        spans.append((int(real_ep_id), int(start), int(end)))
    return spans


def infer_episode_span_from_dataset(dataset: Any, sample_idx: int) -> tuple[int, int, int] | None:
    """Infer `(real_episode_id, episode_start, episode_end)` from a global sample/frame index."""
    for episode_id, start_i, end_i in get_episode_spans_from_dataset(dataset):
        if start_i <= int(sample_idx) < end_i:
            return episode_id, start_i, end_i
    return None


def build_sample_indices_for_labeled_episodes(dataset: Any, labeled_episode_ids: set[int], stride: int) -> list[int]:
    """Build sample indices only from episodes present in the label config."""
    spans = get_episode_spans_from_dataset(dataset)
    if not spans:
        return list(range(0, len(dataset), stride))

    indices: list[int] = []
    available_episode_ids = set()
    for episode_id, start_i, end_i in spans:
        available_episode_ids.add(episode_id)
        if episode_id not in labeled_episode_ids:
            continue
        indices.extend(range(start_i, end_i, stride))

    missing = sorted(labeled_episode_ids - available_episode_ids)
    if missing:
        logger.warning(
            "Label config contains %d episode ids not loaded in dataset, first missing ids: %s",
            len(missing),
            missing[:20],
        )
    logger.info(
        "Collecting %d samples from %d labeled episodes loaded in dataset",
        len(indices),
        len(labeled_episode_ids & available_episode_ids),
    )
    return indices


def load_task_name_map(data_root: str) -> dict[int, str]:
    meta_path = Path(data_root) / "meta" / "tasks.jsonl"
    task_map: dict[int, str] = {}
    if not meta_path.exists():
        return task_map
    for line in meta_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_map[int(row["task_index"])] = str(row["task"])
    return task_map


def normalize_obs_to_model_input(sample: dict[str, Any], processor: FastWAMProcessor, device: str, dtype: torch.dtype):
    img = sample.get("image")
    if img is None:
        video = sample["video"]
        if isinstance(video, torch.Tensor):
            if video.ndim == 4 and video.shape[0] == 3:
                img = video[:, 0]  # [3, H, W]
            elif video.ndim == 4 and video.shape[-1] == 3:
                img = video[0].permute(2, 0, 1)  # [3, H, W]
            elif video.ndim == 5 and video.shape[1] == 3:
                img = video[0, :, 0]  # [3, H, W] from first camera
            elif video.ndim == 5 and video.shape[-1] == 3:
                img = video[0, 0].permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported video tensor shape: {tuple(video.shape)}")
        else:
            raise ValueError("Sample has no `image` and `video` is not a tensor")
    if isinstance(img, torch.Tensor):
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0).cpu().numpy()
        else:
            img = img.cpu().numpy()
    if img.ndim == 4 and img.shape[0] == 1:
        img = img.squeeze(0)
    if img.ndim == 3 and img.shape[0] == 9:
        img = img[:3]
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.asarray(np.clip(img, 0, 255), dtype=np.uint8)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {img.shape}")
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    if x.max() > 1.5:
        x = x * (2.0 / 255.0) - 1.0
    state_key = processor.shape_meta["state"][0]["key"]
    proprio = sample.get("proprio")
    if proprio is None:
        proprio = sample.get("state")
    if proprio is None:
        proprio = np.zeros((processor.shape_meta["state"][0]["shape"][-1],), dtype=np.float32)
    if isinstance(proprio, torch.Tensor):
        proprio = proprio.cpu().numpy()
    proprio = np.asarray(proprio)
    if proprio.ndim == 2:
        proprio = proprio[0]
    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    proprio_t = state_batch["state"][state_key].to(device=device, dtype=dtype)
    prompt = sample.get("prompt") or sample.get("instruction") or DEFAULT_PROMPT.format(task="")
    context = sample.get("context")
    context_mask = sample.get("context_mask")
    return x, proprio_t, prompt, context, context_mask


def _load_cfg_with_hydra(config_path: str):
    cfg_path = Path(config_path).resolve()
    configs_dir = cfg_path.parent.parent if cfg_path.parent.name == "task" else cfg_path.parent
    task_name = cfg_path.stem
    with initialize_config_dir(version_base=None, config_dir=str(configs_dir)):
        cfg = compose(config_name="train", overrides=[f"task={task_name}"])
    return cfg


def main():
    register_default_resolvers()
    args = parse_args()
    setup_logging(log_level=20)

    cfg = _load_cfg_with_hydra(args.config)
    label_cfg = _to_container(args.label_config)
    variant_spec = resolve_label_variant_spec(args.label_variant, label_cfg)
    if variant_spec is None:
        label_keys = infer_label_keys(label_cfg)
        label_maps, label_names = build_label_maps(label_cfg)
        logger.info("Collecting native label keys: %s", label_keys)
    else:
        variant_spec["name"] = args.label_variant
        label_keys = [variant_spec["output_key"]]
        label_maps = variant_spec["label_maps"]
        label_names = variant_spec["label_names"]
        logger.info(
            "Collecting label variant %s: %s -> %s",
            args.label_variant,
            variant_spec["source_key"],
            label_keys[0],
        )
        logger.info("Output label vocab: %s", label_names[label_keys[0]])
    load_text_encoder = bool(OmegaConf.select(cfg, "model.load_text_encoder", default=False))
    print("[probe-debug] cfg.model.load_text_encoder =", OmegaConf.select(cfg, "model.load_text_encoder", default=None))
    print("[probe-debug] resolved use_prompt =", load_text_encoder)

    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=args.device)
    print("[probe-debug] instantiated model type =", type(model))
    if args.inference_mode == "idm" and model.__class__.__name__ != "FastWAMIDM":
        raise TypeError(
            "--inference_mode idm requires the real FastWAMIDM inference path. "
            f"Got model type {type(model)!r}; refusing to fall back to default FastWAM infer_action/infer_joint."
        )
    if hasattr(model, "load_checkpoint"):
        model.load_checkpoint(args.checkpoint)
    else:
        raise TypeError(
            "`cfg.model` did not instantiate into a FastWAM model instance. "
            "Check that the config has a valid `_target_` (for example `fastwam.runtime.create_fastwam`)."
        )
    model.eval()
    model.enable_probe = True

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
        logger.info("Loading all episodes for selected tasks because --max_episodes <= 0")
    dataset = instantiate(OmegaConf.create(dataset_cfg))

    stats_path = cfg.data.train.get("pretrained_norm_stats", None)
    if stats_path is None:
        ckpt_stats = Path(args.checkpoint).resolve().parent / "dataset_stats.json"
        data_root_stats = Path(args.data_root) / "dataset_stats.json"
        if ckpt_stats.exists():
            stats_path = str(ckpt_stats)
        elif data_root_stats.exists():
            stats_path = str(data_root_stats)
        else:
            raise FileNotFoundError(
                "Could not find dataset_stats.json. Checked: "
                f"{ckpt_stats} and {data_root_stats}"
            )
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(stats_path))

    rep_keys = [r.strip() for r in args.representations.split(",") if r.strip()]
    feat_buf = {k: [] for k in rep_keys}
    label_buf = {k: [] for k in label_keys}
    metadata = []

    stride = max(1, args.stride)
    episode_blocks = label_cfg.get("episodes", {}) or label_cfg.get("tasks", {})
    labeled_episode_ids = {int(k) for k in episode_blocks.keys()}
    sample_indices = build_sample_indices_for_labeled_episodes(dataset, labeled_episode_ids, stride)
    if args.dry_run:
        sample_indices = sample_indices[:8]

    skipped_by_variant = 0
    for idx in sample_indices:
        sample = dataset[idx]
        episode_span = infer_episode_span_from_dataset(dataset, idx)
        sample_episode_id = extract_episode_id(sample)
        if episode_span is None:
            if sample_episode_id is None:
                logger.info("Skipping sample %s because episode id could not be determined", idx)
                continue
            episode_key = int(sample_episode_id)
            episode_len = int(sample.get("episode_len", sample.get("length", sample["video"].shape[0])))
            timestep = int(sample.get("timestep", min(idx, episode_len - 1)))
        else:
            episode_key, episode_start, episode_end = episode_span
            episode_len = max(1, int(episode_end) - int(episode_start))
            timestep = int(idx) - int(episode_start)
        episode_blocks = label_cfg.get("episodes", {}) or label_cfg.get("tasks", {})
        if episode_key not in episode_blocks:
            logger.info("Skipping episode %s without label config", episode_key)
            continue

        span = pick_label(episode_blocks[episode_key], timestep, episode_len)
        if variant_spec is None:
            output_label_names = {label_key: span[label_key] for label_key in label_keys}
        else:
            raw_label = span[variant_spec["source_key"]]
            mapped_label = map_variant_label(raw_label, variant_spec)
            if mapped_label is None:
                skipped_by_variant += 1
                continue
            output_label_names = {
                variant_spec["output_key"]: mapped_label,
                f"{variant_spec['source_key']}_raw": raw_label,
            }

        try:
            x, proprio, prompt, context, context_mask = normalize_obs_to_model_input(sample, processor, args.device, model.torch_dtype)
        except Exception as exc:
            logger.warning("Skipping sample %s due to input conversion error: %s", idx, exc)
            continue

        use_prompt = load_text_encoder
        print("[probe-debug] sample idx=", idx, "use_prompt=", use_prompt)
        if use_prompt:
            model.infer_action(
                prompt=prompt,
                input_image=x,
                action_horizon=int(sample.get("action_horizon", 1)),
                num_video_frames=int(sample.get("num_frames", 33)),
                proprio=proprio,
                seed=0,
                num_inference_steps=1,
            )
        else:
            if context is None or context_mask is None:
                logger.warning("Skipping sample %s because prompt context is missing", idx)
                continue
            if isinstance(context, torch.Tensor) and context.ndim == 2:
                context = context.unsqueeze(0)
            if isinstance(context_mask, torch.Tensor) and context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            model.infer_action(
                prompt=None,
                input_image=x,
                action_horizon=int(sample.get("action_horizon", 1)),
                num_video_frames=int(sample.get("num_frames", 33)),
                context=context,
                context_mask=context_mask,
                proprio=proprio,
                seed=0,
                num_inference_steps=1,
            )
        feats = model.get_probe_features()
        if not feats:
            logger.warning("No probe features captured for sample idx=%s", idx)
            continue

        missing_reps = [rep for rep in rep_keys if rep not in feats]
        if missing_reps:
            logger.warning("Skipping sample %s because representations are missing: %s", idx, missing_reps)
            continue
        if args.dry_run or args.print_feature_shapes:
            logger.info("Probe feature shapes for sample %s: %s", idx, {k: tuple(feats[k].shape) for k in rep_keys})

        for rep in rep_keys:
            feat_buf[rep].append(feats[rep].squeeze(0).clone())
        for label_key in label_buf:
            label_buf[label_key].append(
                torch.tensor(label_maps[label_key][output_label_names[label_key]], dtype=torch.long)
            )
        meta = {
            "episode_id": episode_key,
            "timestep": timestep,
            "episode_len": episode_len,
            "instruction": prompt,
        }
        for meta_key, meta_value in output_label_names.items():
            meta[f"{meta_key}_name"] = meta_value
        metadata.append(meta)
        if args.dry_run and len(metadata) >= 3:
            break

    n_meta = len(metadata)
    for rep, values in feat_buf.items():
        if not values:
            logger.warning("Representation %s is completely empty in collected features", rep)
        elif len(values) != n_meta:
            raise RuntimeError(f"Feature/metadata length mismatch for {rep}: {len(values)} vs {n_meta}")
    for label_key, values in label_buf.items():
        if len(values) != n_meta:
            raise RuntimeError(f"Label/metadata length mismatch for {label_key}: {len(values)} vs {n_meta}")

    if variant_spec is not None:
        logger.info(
            "Skipped %d samples due to --label_variant %s filtering",
            skipped_by_variant,
            args.label_variant,
        )

    payload = {
        "features": {k: torch.stack(v) if len(v) else torch.empty((0,)) for k, v in feat_buf.items()},
        "labels": {k: torch.stack(v) if len(v) else torch.empty((0,), dtype=torch.long) for k, v in label_buf.items()},
        "metadata": metadata,
        "label_names": label_names,
        "label_variant": args.label_variant,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    logger.info("Saved probe features to %s", args.output)


if __name__ == "__main__":
    main()
