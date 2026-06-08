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
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _to_container(path: str):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def build_label_maps(label_cfg: dict[str, Any]):
    episode_blocks = label_cfg.get("episodes", {})
    if not episode_blocks:
        episode_blocks = label_cfg.get("tasks", {})
    names = {"effect": [], "stage": [], "target": []}
    for block in episode_blocks.values():
        for item in block.get("labels", []):
            for key in names:
                if item[key] not in names[key]:
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


def infer_episode_id_from_dataset(dataset: Any, sample_idx: int) -> int | None:
    """Infer dataset-local episode id from a global sample/frame index.

    RobotVideoDataset returns training samples/windows, not raw episodes. Its returned
    sample may not contain episode metadata, so we recover the episode by locating
    sample_idx in episode_data_index.
    """
    base_dataset = getattr(dataset, "lerobot_dataset", None)
    episode_data_index = getattr(base_dataset, "episode_data_index", None)
    if episode_data_index is None:
        return None

    starts = episode_data_index.get("from")
    ends = episode_data_index.get("to")
    if starts is None or ends is None:
        return None

    if isinstance(starts, torch.Tensor):
        starts = starts.cpu().numpy()
    if isinstance(ends, torch.Tensor):
        ends = ends.cpu().numpy()

    for ep_idx, (start, end) in enumerate(zip(starts, ends)):
        if int(start) <= int(sample_idx) < int(end):
            return int(ep_idx)
    return None


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
    label_maps, label_names = build_label_maps(label_cfg)
    load_text_encoder = bool(OmegaConf.select(cfg, "model.load_text_encoder", default=False))
    print("[probe-debug] cfg.model.load_text_encoder =", OmegaConf.select(cfg, "model.load_text_encoder", default=None))
    print("[probe-debug] resolved use_prompt =", load_text_encoder)

    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=args.device)
    print("[probe-debug] instantiated model type =", type(model))
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
    dataset_cfg["max_episodes_per_task"] = args.max_episodes
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

    rep_keys = ["video_pre", "video_out", "action_pre", "action_out", "context", "proprio_embed"]
    feat_buf = {k: [] for k in rep_keys}
    label_buf = {k: [] for k in ("effect", "stage", "target")}
    metadata = []

    max_items = 8 if args.dry_run else len(dataset)
    stride = max(1, args.stride)

    for idx in range(0, max_items, stride):
        sample = dataset[idx]
        episode_id = extract_episode_id(sample)
        if episode_id is None:
            episode_id = infer_episode_id_from_dataset(dataset, idx)
        episode_blocks = label_cfg.get("episodes", {}) or label_cfg.get("tasks", {})
        if episode_id is None:
            logger.info("Skipping sample %s because episode id could not be determined", idx)
            continue
        episode_key = int(episode_id)
        if episode_key not in episode_blocks:
            logger.info("Skipping episode %s without label config", episode_key)
            continue
        episode_len = int(sample.get("episode_len", sample.get("length", sample["video"].shape[0])))
        timestep = int(sample.get("timestep", min(idx, episode_len - 1)))
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

        span = pick_label(episode_blocks[episode_key], timestep, episode_len)
        for rep in rep_keys:
            if rep in feats:
                feat_buf[rep].append(feats[rep].squeeze(0).clone())
        for label_key in label_buf:
            label_buf[label_key].append(torch.tensor(label_maps[label_key][span[label_key]], dtype=torch.long))
        metadata.append({
            "episode_id": episode_key,
            "timestep": timestep,
            "episode_len": episode_len,
            "instruction": prompt,
            "effect_name": span["effect"],
            "stage_name": span["stage"],
            "target_name": span["target"],
        })
        if args.dry_run and len(metadata) >= 3:
            break

    payload = {
        "features": {k: torch.stack(v) if len(v) else torch.empty((0,)) for k, v in feat_buf.items()},
        "labels": {k: torch.stack(v) if len(v) else torch.empty((0,), dtype=torch.long) for k, v in label_buf.items()},
        "metadata": metadata,
        "label_names": label_names,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    logger.info("Saved probe features to %s", args.output)


if __name__ == "__main__":
    main()
