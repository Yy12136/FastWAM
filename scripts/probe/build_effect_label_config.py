import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations_dir", required=True, help="Directory containing episode_*.yaml files")
    p.add_argument("--output", required=True, help="Output label config yaml path")
    p.add_argument("--task_name_mode", choices=["first", "all"], default="first")
    p.add_argument("--include_stage", action="store_true", help="Keep stage field from annotations")
    return p.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def normalize_task_name(task_field: Any) -> str:
    if isinstance(task_field, list):
        return str(task_field[0]) if task_field else ""
    return str(task_field or "")


def main():
    args = parse_args()
    annotations_dir = Path(args.annotations_dir)
    output_path = Path(args.output)

    episode_files = sorted(annotations_dir.glob("episode_*.yaml"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.yaml files found in {annotations_dir}")

    episodes: dict[int, dict[str, Any]] = {}
    effect_order = []
    target_order = []
    stage_order = []

    for path in episode_files:
        ann = load_yaml(path)
        episode_index = int(ann.get("episode_index"))
        task_name = normalize_task_name(ann.get("task"))
        segments = ann.get("segments", []) or []
        if not segments:
            continue

        total_end = max(int(seg["end_frame"]) for seg in segments)
        denom = max(total_end, 1)
        labels = []
        for seg in segments:
            start_frame = int(seg["start_frame"])
            end_frame = int(seg["end_frame"])
            effect = str(seg.get("effect", "other"))
            target = str(seg.get("target", ""))
            stage = str(seg.get("stage", "other"))

            label = {
                "start_ratio": float(start_frame) / float(denom),
                "end_ratio": float(end_frame) / float(denom),
                "effect": effect,
                "target": target,
            }
            label["stage"] = stage if args.include_stage else "other"
            labels.append(label)

            if effect not in effect_order:
                effect_order.append(effect)
            if target not in target_order:
                target_order.append(target)
            if stage not in stage_order:
                stage_order.append(stage)

        episodes[episode_index] = {
            "task_name": task_name,
            "labels": labels,
        }

    payload = {
        "episodes": episodes,
        "label_vocab": {
            "effect": effect_order,
            "target": target_order,
            "stage": stage_order if args.include_stage else ["other"],
        },
        "source_annotations_dir": str(annotations_dir),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    print(f"Wrote label config to {output_path}")
    print(f"Episodes: {len(episodes)}")
    print(f"Effects: {effect_order}")
    print(f"Targets: {target_order}")


if __name__ == "__main__":
    main()
