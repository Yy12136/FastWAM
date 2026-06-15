import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


DEFAULT_STAGE_PROMPT = """你是一个机器人任务视频标注助手。
请根据任务 instruction 和输入关键帧，为当前 episode 做阶段、效果、目标物体的自动标注。

要求：
1. 只输出严格 JSON，不要输出多余解释。
2. 你的标注目标是把整个 episode 切成若干连续阶段。
3. 每个阶段包含：start_frame, end_frame, stage, effect, target, confidence, rationale。
4. stage 只能从下面枚举中选择：
   - approach
   - grasp
   - move
   - place
   - open
   - close
   - press
   - release
   - pickup
   - putdown
   - other
5. effect 表示这个阶段对目标造成的语义变化，要优先从 instruction 里找描述，再结合视频判断。
   - effect 必须使用 snake_case，不允许空格。
   - effect 应该尽量包含 target 和状态变化，例如 stove_off_to_on、moka_pot_on_table_to_on_stove、drawer_closed_to_open、object_on_table_to_in_basket。
   - 如果该阶段只是接近、移动手臂、尚未造成明确状态变化，可以写 none 或 unknown。
6. target 表示这个阶段主要作用的物体，也必须使用 snake_case，不允许空格，例如 stove、moka_pot、drawer、basket。
7. 如果 instruction 是 "turn on the stove and put the moka pot on it"：
   - 看到 stove 被打开时，effect 应为 stove_off_to_on，target 应为 stove。
   - 看到 moka pot 被放到 stove 上时，effect 应为 moka_pot_not_on_stove_to_on_stove 或 moka_pot_on_table_to_on_stove，target 应为 moka_pot。
8. 如果无法判断，stage 使用 other，effect 使用 unknown，target 使用 unknown。
9. start_frame 和 end_frame 使用输入帧中的 frame_idx。
10. 保证阶段按时间排序且不重叠，最后一个阶段的 end_frame 必须等于 episode 最后一帧。
11. 所有标签字段 stage/effect/target 都必须是小写 snake_case，不允许空格、连字符或标点。

输出 JSON 格式：
{
  "task": "...",
  "episode_index": 0,
  "segments": [
    {
      "start_frame": 0,
      "end_frame": 12,
      "stage": "approach",
      "effect": "none",
      "target": "stove",
      "confidence": 0.78,
      "rationale": "..."
    }
  ]
}
"""


@dataclass
class FrameSample:
    frame_idx: int
    image_path: str


@dataclass
class EpisodeAnnotation:
    task: str
    episode_index: int
    episode_id: int
    video_path: str
    segments: list[dict[str, Any]]
    model_name: str
    raw_response: str


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="e.g. ./data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot")
    p.add_argument("--output_root", required=True, help="Directory to store per-task json files")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--frame_count", type=int, default=12)
    p.add_argument("--max_pixels", type=int, default=512 * 512, help="Resize sampled frames before sending to Qwen")
    p.add_argument("--episode_limit", type=int, default=-1, help="Optional limit for debugging")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_video_path(data_root: Path, info: dict[str, Any], episode_index: int, episode: dict[str, Any]) -> Path:
    pattern = info["video_path"]
    episode_chunk = episode_index // int(info.get("chunks_size", 1000))
    video_key = "observation.images.image"
    rel = pattern.format(episode_chunk=episode_chunk, video_key=video_key, episode_index=episode_index)
    return data_root / rel


def build_task_lookup(tasks: list[dict[str, Any]]) -> tuple[dict[int, str], dict[str, str]]:
    by_index: dict[int, str] = {}
    by_normalized_text: dict[str, str] = {}
    for task in tasks:
        instruction = task.get("task") or task.get("name") or task.get("language_instruction") or task.get("instruction")
        if not instruction:
            continue
        instruction = str(instruction).strip()
        task_index = task.get("task_index")
        if task_index is not None:
            by_index[int(task_index)] = instruction
        by_normalized_text[to_snake_label(instruction)] = instruction
    return by_index, by_normalized_text


def get_episode_instruction(
    episode: dict[str, Any],
    task_by_index: dict[int, str],
    task_by_normalized_text: dict[str, str],
) -> str:
    if "task_index" in episode and int(episode["task_index"]) in task_by_index:
        return task_by_index[int(episode["task_index"])]
    if "task_indices" in episode and episode["task_indices"]:
        first_index = int(episode["task_indices"][0])
        if first_index in task_by_index:
            return task_by_index[first_index]
    tasks = episode.get("tasks", [])
    if isinstance(tasks, list) and tasks:
        raw_task = str(tasks[0]).strip()
    else:
        raw_task = str(tasks).strip() if tasks else "unknown"
    return task_by_normalized_text.get(to_snake_label(raw_task), raw_task)


def extract_frame_samples(video_path: Path, frame_count: int, max_pixels: int) -> list[FrameSample]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    frame_ids = sorted(set(int(round(i)) for i in np.linspace(0, total - 1, frame_count)))
    samples: list[FrameSample] = []
    tmp_dir = video_path.parent / "qwen25vl_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for idx in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        if h * w > max_pixels:
            scale = (max_pixels / float(h * w)) ** 0.5
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        out_path = tmp_dir / f"{video_path.stem}_frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        samples.append(FrameSample(frame_idx=idx, image_path=str(out_path)))
    cap.release()
    return samples


def build_messages(task_name: str, episode_index: int, video_path: Path, frame_samples: list[FrameSample], instruction: str):
    prompt = [
        DEFAULT_STAGE_PROMPT.strip(),
        f"task: {task_name}",
        f"instruction: {instruction}",
        f"episode_index: {episode_index}",
        f"video_path: {video_path}",
        "frames:",
    ]
    for s in frame_samples:
        prompt.append(f"- frame_idx={s.frame_idx}")
    content = [{"type": "text", "text": "\n".join(prompt)}]
    for s in frame_samples:
        content.append({"type": "image", "image": s.image_path})
    return [{"role": "user", "content": content}]


def get_torch_dtype(dtype_name: str):
    import torch
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def parse_model_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end >= 0:
        raw = raw[start:end + 1]
    return json.loads(raw)


def to_snake_label(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def normalize_segments(payload: dict[str, Any], episode_len: int):
    segments = payload.get("segments", [])
    cleaned = []
    last_end = -1
    for seg in segments:
        start = int(seg.get("start_frame", 0))
        end = int(seg.get("end_frame", 0))
        start = max(0, min(start, episode_len - 1))
        end = max(start, min(end, episode_len - 1))
        if start <= last_end:
            start = last_end + 1
        if start > end:
            continue
        cleaned.append({
            "start_frame": start,
            "end_frame": end,
            "stage": to_snake_label(seg.get("stage", "other"), default="other"),
            "effect": to_snake_label(seg.get("effect", "unknown"), default="unknown"),
            "target": to_snake_label(seg.get("target", "unknown"), default="unknown"),
            "confidence": float(seg.get("confidence", 0.0)),
            "rationale": str(seg.get("rationale", "")),
        })
        last_end = end
    if cleaned:
        cleaned[-1]["end_frame"] = episode_len - 1
    return cleaned


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    info = json.loads((data_root / "meta" / "info.json").read_text())
    episodes = load_jsonl(data_root / "meta" / "episodes.jsonl")
    tasks = load_jsonl(data_root / "meta" / "tasks.jsonl")
    task_by_index, task_by_normalized_text = build_task_lookup(tasks)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import torch

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=get_torch_dtype(args.dtype),
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model.eval()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    episode_total = len(episodes) if args.episode_limit < 0 else min(args.episode_limit, len(episodes))

    for episode_index in range(episode_total):
        episode = episodes[episode_index]
        instruction = get_episode_instruction(episode, task_by_index, task_by_normalized_text)
        task_name = instruction
        grouped[task_name].append({"episode_index": episode_index, "episode": episode, "instruction": instruction})

    for task_name, eps in grouped.items():
        task_output = output_root / f"{to_snake_label(task_name, default='task')}.json"
        if task_output.exists() and not args.overwrite:
            print(f"Skip existing: {task_output}")
            continue

        task_payload = {
            "task": task_name,
            "task_key": to_snake_label(task_name, default="task"),
            "model_name": args.model_id,
            "data_root": str(data_root),
            "episodes": {},
        }

        for item in eps:
            episode_index = item["episode_index"]
            episode = item["episode"]
            instruction = item["instruction"]
            video_path = resolve_video_path(data_root, info, episode_index, episode)
            frame_samples = extract_frame_samples(video_path, args.frame_count, args.max_pixels)
            if not frame_samples:
                print(f"No frames extracted for episode {episode_index}, skip")
                continue

            messages = build_messages(task_name, episode_index, video_path, frame_samples, instruction)
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs = [Image.open(s.image_path).convert("RGB") for s in frame_samples]
            inputs = processor(text=[prompt], images=image_inputs, return_tensors="pt")
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=1024)
                input_token_len = inputs["input_ids"].shape[-1]
                generated_ids = generated_ids[:, input_token_len:]
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            try:
                parsed = parse_model_json(generated_text)
            except Exception:
                parsed = {
                    "task": task_name,
                    "episode_index": episode_index,
                    "segments": [],
                    "parse_error": True,
                }

            episode_len = int(episode.get("length", frame_samples[-1].frame_idx + 1))
            cleaned = normalize_segments(parsed, episode_len)
            annotation = EpisodeAnnotation(
                task=task_name,
                episode_index=episode_index,
                episode_id=episode.get("episode_index", episode_index),
                video_path=str(video_path),
                segments=cleaned,
                model_name=args.model_id,
                raw_response=generated_text,
            )
            task_payload["episodes"][str(episode_index)] = asdict(annotation)
            print(f"Labeled task={task_name} episode={episode_index} segments={len(cleaned)}")

        task_output.write_text(json.dumps(task_payload, ensure_ascii=False, indent=2))
        print(f"Saved {task_output}")


if __name__ == "__main__":
    main()
