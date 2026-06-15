from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam_joint import FastWAMJoint

logger = get_logger(__name__)


class FastWAMIDM(FastWAMJoint):
    """IDM variant with teacher-forcing video conditioning for action denoising."""

    # Hardcoded probability: during training, cond-video is noised with this chance.
    video_cond_noise_prob = 0.5

    def __init__(self, *args, use_factor_aux_loss: bool = False, lambda_effect: float = 0.05, lambda_target: float = 0.05, factor_aux_num_effect_classes: int = 1, factor_aux_num_target_classes: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_factor_aux_loss = bool(use_factor_aux_loss)
        self.lambda_effect = float(lambda_effect)
        self.lambda_target = float(lambda_target)
        self.factor_aux_num_effect_classes = int(factor_aux_num_effect_classes)
        self.factor_aux_num_target_classes = int(factor_aux_num_target_classes)
        hidden_dim = int(getattr(self, "text_dim", 4096))
        self.effect_head = nn.Linear(hidden_dim, self.factor_aux_num_effect_classes)
        self.target_head = nn.Linear(hidden_dim, self.factor_aux_num_target_classes)
        self.factor_aux_label_map = {"effect": {}, "target": {}}

    def set_factor_aux_label_map(self, factor_aux_label_map: dict[str, dict[str, int]]):
        self.factor_aux_label_map = factor_aux_label_map
        effect_classes = max(1, len(factor_aux_label_map.get("effect", {})))
        target_classes = max(1, len(factor_aux_label_map.get("target", {})))
        if self.effect_head.out_features != effect_classes:
            self.effect_head = nn.Linear(self.effect_head.in_features, effect_classes).to(self.device, dtype=self.torch_dtype)
            self.factor_aux_num_effect_classes = effect_classes
        if self.target_head.out_features != target_classes:
            self.target_head = nn.Linear(self.target_head.in_features, target_classes).to(self.device, dtype=self.torch_dtype)
            self.factor_aux_num_target_classes = target_classes

    def _ensure_factor_aux_heads(self, input_dim: int):
        input_dim = int(input_dim)
        if self.effect_head.in_features != input_dim:
            self.effect_head = nn.Linear(
                input_dim,
                self.factor_aux_num_effect_classes,
            ).to(self.device, dtype=self.torch_dtype)
        if self.target_head.in_features != input_dim:
            self.target_head = nn.Linear(
                input_dim,
                self.factor_aux_num_target_classes,
            ).to(self.device, dtype=self.torch_dtype)

    def _factor_aux_loss(self, action_repr: torch.Tensor, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.use_factor_aux_loss:
            return action_repr.new_zeros(()), {}
        if action_repr.ndim == 3:
            pooled = action_repr.mean(dim=1)
        else:
            pooled = action_repr
        self._ensure_factor_aux_heads(pooled.shape[-1])
        effect_logits = self.effect_head(pooled.to(dtype=self.effect_head.weight.dtype))
        target_logits = self.target_head(pooled.to(dtype=self.target_head.weight.dtype))
        effect_label = sample.get("factor_aux_effect_label")
        target_label = sample.get("factor_aux_target_label")
        loss = pooled.new_zeros(())
        loss_dict = {}
        if effect_label is not None:
            effect_label = effect_label.to(device=effect_logits.device, dtype=torch.long)
            valid = effect_label.ge(0)
            if bool(valid.any()):
                loss_effect = F.cross_entropy(effect_logits[valid], effect_label[valid])
                loss = loss + self.lambda_effect * loss_effect
                loss_dict["loss_effect"] = float((self.lambda_effect * loss_effect).detach().item())
        if target_label is not None:
            target_label = target_label.to(device=target_logits.device, dtype=torch.long)
            valid = target_label.ge(0)
            if bool(valid.any()):
                loss_target = F.cross_entropy(target_logits[valid], target_label[valid])
                loss = loss + self.lambda_target * loss_target
                loss_dict["loss_target"] = float((self.lambda_target * loss_target).detach().item())
        return loss, loss_dict

    @torch.no_grad()
    def _build_teacher_forcing_attention_mask(
        self,
        noisy_video_seq_len: int,
        cond_video_seq_len: int,
        action_seq_len: int,
        noisy_video_tokens_per_frame: int,
        cond_video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if noisy_video_tokens_per_frame != cond_video_tokens_per_frame:
            raise ValueError(
                "Teacher-forcing requires identical `tokens_per_frame` for noisy and cond video branches, "
                f"got {noisy_video_tokens_per_frame} and {cond_video_tokens_per_frame}."
            )

        noisy_end = noisy_video_seq_len
        cond_end = noisy_video_seq_len + cond_video_seq_len
        total_seq_len = cond_end + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:noisy_end, :noisy_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=noisy_video_seq_len,
            video_tokens_per_frame=noisy_video_tokens_per_frame,
            device=device,
        )
        mask[noisy_end:cond_end, noisy_end:cond_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=cond_video_seq_len,
            video_tokens_per_frame=cond_video_tokens_per_frame,
            device=device,
        )
        mask[cond_end:, cond_end:] = True
        mask[cond_end:, noisy_end:cond_end] = True
        return mask

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(batch_size=batch_size, device=self.device, dtype=input_latents.dtype)
        latents_noisy = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents_noisy[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(batch_size=batch_size, device=self.device, dtype=action.dtype)
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        cond_noise_mask = torch.rand((batch_size,), device=self.device) < float(self.video_cond_noise_prob)
        timestep_video_cond = torch.zeros_like(timestep_video, dtype=input_latents.dtype, device=self.device)
        latents_cond = input_latents
        if bool(cond_noise_mask.any()):
            timestep_video_cond_sampled = self.train_video_scheduler.sample_training_t(batch_size=batch_size, device=self.device, dtype=input_latents.dtype)
            timestep_video_cond = torch.where(cond_noise_mask, timestep_video_cond_sampled, timestep_video_cond)
            noise_video_cond = torch.randn_like(input_latents)
            latents_cond_noisy = self.train_video_scheduler.add_noise(input_latents, noise_video_cond, timestep_video_cond_sampled)
            latents_cond = torch.where(cond_noise_mask.view(batch_size, 1, 1, 1, 1), latents_cond_noisy, input_latents)
        if inputs["first_frame_latents"] is not None:
            latents_cond = latents_cond.clone()
            latents_cond[:, :, 0:1] = inputs["first_frame_latents"]

        video_pre_noisy = self.video_expert.pre_dit(x=latents_noisy, timestep=timestep_video, context=context, context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse_flag)
        video_pre_cond = self.video_expert.pre_dit(x=latents_cond, timestep=timestep_video_cond, context=context, context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse_flag)
        action_pre = self.action_expert.pre_dit(action_tokens=noisy_action, timestep=timestep_action, context=context, context_mask=context_mask)

        merged_video_tokens = torch.cat([video_pre_noisy["tokens"], video_pre_cond["tokens"]], dim=1)
        merged_video_freqs = torch.cat([video_pre_noisy["freqs"], video_pre_cond["freqs"]], dim=0)
        merged_video_t_mod = torch.cat([video_pre_noisy["t_mod"], video_pre_cond["t_mod"]], dim=1)
        merged_video_context_mask = torch.cat([video_pre_noisy["context_mask"], video_pre_cond["context_mask"]], dim=1)
        attention_mask = self._build_teacher_forcing_attention_mask(
            noisy_video_seq_len=int(video_pre_noisy["tokens"].shape[1]),
            cond_video_seq_len=int(video_pre_cond["tokens"].shape[1]),
            action_seq_len=action_pre["tokens"].shape[1],
            noisy_video_tokens_per_frame=int(video_pre_noisy["meta"]["tokens_per_frame"]),
            cond_video_tokens_per_frame=int(video_pre_cond["meta"]["tokens_per_frame"]),
            device=merged_video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={"video": merged_video_tokens, "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": merged_video_freqs, "action": action_pre["freqs"]},
            context_all={"video": {"context": video_pre_noisy["context"], "mask": merged_video_context_mask}, "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]}},
            t_mod_all={"video": merged_video_t_mod, "action": action_pre["t_mod"]},
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"][:, : int(video_pre_noisy["tokens"].shape[1])], video_pre_noisy)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(pred_video=pred_video, target_video=target_video, image_is_pad=image_is_pad, include_initial_video_step=include_initial_video_step)
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(loss_video_per_sample.device, dtype=loss_video_per_sample.dtype)
        loss_video = (loss_video_per_sample * video_weight).mean()
        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(action_loss_per_sample.device, dtype=action_loss_per_sample.dtype)
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_factor, factor_loss_dict = self._factor_aux_loss(action_pre["tokens"], sample)
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action + loss_factor
        loss_dict = {"loss_video": self.loss_lambda_video * float(loss_video.detach().item()), "loss_action": self.loss_lambda_action * float(loss_action.detach().item())}
        loss_dict.update(factor_loss_dict)
        return loss_total, loss_dict
