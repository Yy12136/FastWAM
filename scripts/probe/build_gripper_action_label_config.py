import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


GRIPPER_VOCAB = ["open", "close", "static"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Build open/close/static gripper-action pseudo labels from LeRobot gripper state."
    )
    p.add_argument("--data_root", required=True, help="LeRobot dataset root")
    p.add_argument(
        "--episode_source_labels",
        required=True,
        help="YAML label config whose episode ids select the episodes to process",
    )
    p.add_argument("--output", required=True, help="Output gripper_action label config yaml")
    p.add_argument("--threshold", type=float, default=0.01, help="Gripper delta threshold")
    p.add_argument("--smooth_window", type=int, default=3, help="Centered smoothing window for gripper state")
    p.add_argument("--delta_window", type=int, default=3, help="Frame offset used to compute gripper delta")
    p.add_argument("--min_segment_len", type=int, default=3, help="Merge segments shorter than this many frames")
    p.add_argument("--invert", action="store_true", help="Invert open/close direction if dataset convention is opposite")
    return p.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_tasks(data_root: Path) -> dict[int, str]:
    path = data_root / "meta" / "tasks.jsonl"
    tasks: dict[int, str] = {}
    if not path.exists():
        return tasks
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tasks[int(row["task_index"])] = str(row["task"])
    return tasks


def load_info(data_root: Path) -> dict[str, Any]:
    with open(data_root / "meta" / "info.json", "r") as f:
        return json.load(f)


def episode_parquet_path(data_root: Path, info: dict[str, Any], episode_id: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_id // chunks_size
    rel = info["data_path"].format(episode_chunk=episode_chunk, episode_index=episode_id)
    return data_root / rel


def _stack_series(values: np.ndarray) -> np.ndarray:
    rows = []
    for value in values:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        rows.append(arr)
    return np.stack(rows).astype(np.float32)


def extract_gripper_signal(df: pd.DataFrame, prefer_action: bool) -> tuple[np.ndarray, str, str]:
    candidates = ["action", "observation.states.gripper_state", "observation.state"] if prefer_action else [
        "observation.states.gripper_state",
        "observation.state",
        "action",
    ]
    for col in candidates:
        if col not in df.columns:
            continue
        arr = _stack_series(df[col].to_numpy())
        if arr.ndim != 2 or arr.shape[1] == 0:
            continue
        if col == "action":
            return arr[:, -1].astype(np.float32), col, "command"
        if col in {"observation.states.gripper_state", "observation.state"} and arr.shape[1] >= 2:
            # Franka gripper state is stored as two opposing finger coordinates.
            # Mean cancels the signal, so use aperture between the last two dims.
            return np.abs(arr[:, -2] - arr[:, -1]).astype(np.float32), col, "aperture"
        return arr[:, -1].astype(np.float32), col, "absolute"
    raise KeyError(
        "Could not find gripper signal column. Tried: " + ", ".join(candidates)
    )


def smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1 or len(x) <= 2:
        return x.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(x.astype(np.float32), (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def classify_gripper(gripper: np.ndarray, threshold: float, delta_window: int, invert: bool) -> list[str]:
    n = len(gripper)
    delta_window = max(1, int(delta_window))
    labels: list[str] = []
    for i in range(n):
        j = max(0, i - delta_window)
        delta = float(gripper[i] - gripper[j])
        if invert:
            delta = -delta
        if delta > threshold:
            labels.append("open")
        elif delta < -threshold:
            labels.append("close")
        else:
            labels.append("static")
    return labels


def segments_from_labels(labels: list[str], min_segment_len: int) -> list[tuple[int, int, str]]:
    if not labels:
        return []
    segments: list[tuple[int, int, str]] = []
    start = 0
    current = labels[0]
    for i, label in enumerate(labels[1:], start=1):
        if label != current:
            segments.append((start, i, current))
            start = i
            current = label
    segments.append((start, len(labels), current))

    min_segment_len = max(1, int(min_segment_len))
    if min_segment_len <= 1 or len(segments) <= 1:
        return segments

    merged: list[tuple[int, int, str]] = []
    for seg in segments:
        start_i, end_i, label = seg
        if merged and (end_i - start_i) < min_segment_len:
            prev_start, _, prev_label = merged[-1]
            merged[-1] = (prev_start, end_i, prev_label)
        else:
            merged.append(seg)

    # Merge adjacent equal labels created by short-segment absorption.
    compact: list[tuple[int, int, str]] = []
    for start_i, end_i, label in merged:
        if compact and compact[-1][2] == label:
            prev_start, _, _ = compact[-1]
            compact[-1] = (prev_start, end_i, label)
        else:
            compact.append((start_i, end_i, label))
    return compact


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    source_cfg = load_yaml(args.episode_source_labels)
    episode_blocks = source_cfg.get("episodes", {}) or source_cfg.get("tasks", {})
    episode_ids = sorted(int(k) for k in episode_blocks.keys())
    if not episode_ids:
        raise ValueError(f"No episodes found in {args.episode_source_labels}")

    info = load_info(data_root)
    tasks = load_tasks(data_root)
    episodes: dict[int, dict[str, Any]] = {}

    for episode_id in episode_ids:
        parquet_path = episode_parquet_path(data_root, info, episode_id)
        if not parquet_path.exists():
            print(f"[WARN] Missing episode parquet for episode {episode_id}: {parquet_path}")
            continue
        df = pd.read_parquet(parquet_path)
        gripper, source_column, signal_type = extract_gripper_signal(df, prefer_action=False)
        gripper = smooth_1d(gripper, args.smooth_window)
        labels = classify_gripper(
            gripper=gripper,
            threshold=float(args.threshold),
            delta_window=int(args.delta_window),
            invert=bool(args.invert),
        )
        segments = segments_from_labels(labels, args.min_segment_len)
        n = max(1, len(labels))

        task_name = ""
        if "task_index" in df.columns and len(df) > 0:
            task_index = int(df["task_index"].iloc[0])
            task_name = tasks.get(task_index, "")
        if not task_name:
            task_name = str(episode_blocks.get(episode_id, {}).get("task_name", ""))

        label_items = []
        for start_i, end_i, label in segments:
            label_items.append(
                {
                    "start_ratio": float(start_i) / float(n),
                    "end_ratio": float(end_i) / float(n),
                    "gripper_action": label,
                    "start_frame": int(start_i),
                    "end_frame": int(end_i),
                }
            )
        if label_items:
            label_items[0]["start_ratio"] = 0.0
            label_items[-1]["end_ratio"] = 1.0

        episodes[int(episode_id)] = {
            "task_name": task_name,
            "labels": label_items,
            "num_frames": int(n),
            "source": "observation.states.gripper_state_or_fallback",
        }

    payload = {
        "episodes": episodes,
        "label_vocab": {"gripper_action": GRIPPER_VOCAB},
        "source_episode_labels": str(args.episode_source_labels),
        "data_root": str(data_root),
        "params": {
            "threshold": float(args.threshold),
            "smooth_window": int(args.smooth_window),
            "delta_window": int(args.delta_window),
            "min_segment_len": int(args.min_segment_len),
            "invert": bool(args.invert),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    counts = {k: 0 for k in GRIPPER_VOCAB}
    for block in episodes.values():
        for item in block["labels"]:
            counts[item["gripper_action"]] += 1
    print(f"Wrote gripper-action labels to {output_path}")
    print(f"Episodes: {len(episodes)}")
    print(f"Segment counts: {counts}")


if __name__ == "__main__":
    main()
