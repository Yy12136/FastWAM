import logging
import os
import inspect
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf
import yaml

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
from .utils.video_io import save_mp4
from .utils import misc

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _load_factor_annotation_map(annotation_path: str | None, output_dir: str | None = None):
    if not annotation_path:
        return {}, {}
    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    episodes = payload.get("episodes", {})
    effect_map, target_map = {}, {}
    for ep_val in episodes.values():
        for label in ep_val.get("labels", []):
            effect_map.setdefault(label["effect"], len(effect_map))
            target_map.setdefault(label["target"], len(target_map))
    if output_dir:
        out = Path(output_dir) / "factor_aux_label_map.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump({"effect": effect_map, "target": target_map}, f, sort_keys=True)
    return {"effect": effect_map, "target": target_map}, episodes


def _instantiate_optional_dataset(data_cfg: DictConfig, split: str):
    if split not in data_cfg or data_cfg.get(split) is None:
        return None
    return instantiate(data_cfg[split])


def _to_plain_container(value):
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def run_training(cfg: DictConfig):
    """Instantiate configured datasets/model/trainer and run training."""
    from accelerate import PartialState

    distributed_state = PartialState()
    setup_logging(is_main_process=distributed_state.is_main_process)
    logger.info("Training config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.get("data") is None or cfg.data.get("train") is None:
        raise ValueError("Training config must define `data.train`.")
    if cfg.get("model") is None:
        raise ValueError("Training config must define `model`.")

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_kwargs = {
        "model_dtype": _mixed_precision_to_model_dtype(cfg.mixed_precision),
        "device": str(distributed_state.device),
    }

    # Backward-compatible root-level CLI aliases. This lets existing commands such as
    # `use_factor_aux_loss=true lambda_effect=0.05` keep working while the actual
    # constructor arguments still live under `model.*`.
    for key in ("use_factor_aux_loss", "factor_annotation_path", "lambda_effect", "lambda_target"):
        if key in model_cfg and model_cfg.get(key) is not None:
            model_kwargs[key] = model_cfg.get(key)
        if key in cfg and cfg.get(key) is not None:
            model_kwargs[key] = cfg.get(key)

    factor_annotation_path = model_kwargs.get("factor_annotation_path", None)
    factor_label_map, episodes = _load_factor_annotation_map(factor_annotation_path)

    train_dataset = instantiate(cfg.data.train)
    val_dataset = _instantiate_optional_dataset(cfg.data, "val")
    if hasattr(train_dataset, "set_factor_aux_labels"):
        train_dataset.set_factor_aux_labels(factor_label_map, episodes)
    if val_dataset is not None and hasattr(val_dataset, "set_factor_aux_labels"):
        val_dataset.set_factor_aux_labels(factor_label_map, episodes)
    model = instantiate(model_cfg, **model_kwargs)

    trainer = Wan22Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=cfg,
    )
    trainer.train()
    return trainer


def create_fastwam_idm(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    use_factor_aux_loss: bool = False,
    lambda_effect: float = 0.05,
    lambda_target: float = 0.05,
    factor_annotation_path: str | None = None,
):
    from .models.wan22.fastwam_idm import FastWAMIDM

    video_dit_config = _to_plain_container(video_dit_config)
    action_dit_config = _to_plain_container(action_dit_config)
    video_scheduler = _to_plain_container(video_scheduler) or {}
    action_scheduler = _to_plain_container(action_scheduler) or {}
    loss = _to_plain_container(loss) or {}

    factor_label_map, episodes = _load_factor_annotation_map(factor_annotation_path)
    model = FastWAMIDM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )
    model.use_factor_aux_loss = bool(use_factor_aux_loss)
    model.lambda_effect = float(lambda_effect)
    model.lambda_target = float(lambda_target)
    if use_factor_aux_loss:
        model.set_factor_aux_label_map(factor_label_map)
    return model
