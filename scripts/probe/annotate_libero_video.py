import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import yaml


@dataclass
class Segment:
    start_frame: int
    end_frame: int
    effect: str
    stage: str
    target: str


DEFAULT_EFFECTS = [
    "approach",
    "grasp",
    "move",
    "place",
    "open",
    "close",
    "press",
    "release",
    "pickup",
    "putdown",
    "other",
]

DEFAULT_STAGES = ["start", "middle", "end", "other"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--episode_index", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--video_path", default=None, help="Optional explicit mp4 path")
    p.add_argument("--fps", type=float, default=None, help="Override playback fps")
    p.add_argument("--frame_step", type=int, default=10, help="Frame step for left/right keys")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--max_width", type=int, default=1280)
    return p.parse_args()


def load_meta(data_root: Path):
    meta_dir = data_root / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    episodes = [json.loads(line) for line in (meta_dir / "episodes.jsonl").read_text().splitlines() if line.strip()]
    return info, episodes


def resolve_video_path(data_root: Path, info: dict[str, Any], episode_index: int, explicit: str | None):
    if explicit:
        return Path(explicit)
    pattern = info["video_path"]
    episode_chunk = episode_index // int(info.get("chunks_size", 1000))
    video_key = "observation.images.image"
    rel = pattern.format(
        episode_chunk=episode_chunk,
        video_key=video_key,
        episode_index=episode_index,
    )
    return data_root / rel


def load_video_frames(video_path: Path):
    try:
        import av  # type: ignore

        container = av.open(str(video_path))
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
        if frames:
            return frames
    except Exception:
        pass

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video with OpenCV: {video_path}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if frames:
            return frames
    except Exception:
        pass

    raise RuntimeError(
        f"Unable to decode video: {video_path}. Install `av` (PyAV) or use a video backend with AV1 support."
    )


def draw_overlay(frame, text_lines):
    canvas = frame.copy()
    y = 28
    for line in text_lines:
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        y += 28
    return canvas


def prompt_text(title: str, default: str = "") -> str:
    value = input(f"{title} [{default}]: ").strip()
    return value if value else default


def choose_from_list(title: str, options: list[str], default: str = "other") -> str:
    print(f"\n{title}")
    for i, opt in enumerate(options):
        print(f"  {i}: {opt}")
    raw = input(f"Choose index or type custom [{default}]: ").strip()
    if raw == "":
        return default
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < len(options):
            return options[idx]
    return raw


def save_annotations(output: Path, payload: dict[str, Any]):
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    info, episodes = load_meta(data_root)
    if args.episode_index < 0 or args.episode_index >= len(episodes):
        raise ValueError(f"episode_index out of range: {args.episode_index}")

    ep = episodes[args.episode_index]
    video_path = resolve_video_path(data_root, info, args.episode_index, args.video_path)
    frames = load_video_frames(video_path)
    total_frames = len(frames)

    headless = not bool(os.environ.get("DISPLAY"))
    if headless:
        print("No DISPLAY found; running in terminal annotation mode.")

    print(f"Loaded episode {args.episode_index}")
    print(f"Task: {ep.get('tasks')}")
    print(f"Length: {ep.get('length')} | video frames: {total_frames}")
    print("Controls: space/play-pause | a/d step | j/k jump 10 | n new segment | s save | q quit")

    current = max(0, min(args.start_frame, total_frames - 1))
    playing = False
    segments: list[Segment] = []
    current_segment_start = 0
    current_effect = "other"
    current_stage = "other"
    current_target = ""
    fps = float(args.fps or info.get("fps", 20))
    delay = max(1, int(1000 / fps))

    preview_dir = Path(args.output).with_suffix("") / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    while True:
        frame = frames[current]
        overlay = [
            f"episode={args.episode_index} frame={current}/{total_frames - 1}",
            f"segment_start={current_segment_start} effect={current_effect} stage={current_stage} target={current_target or '-'}",
            f"keys: space play/pause | a/d step | j/k jump | n mark_end | e edit labels | s save | q quit",
        ]
        canvas = draw_overlay(frame, overlay)

        if headless:
            preview_path = preview_dir / f"frame_{current:06d}.png"
            cv2.imwrite(str(preview_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            print("\n" + " | ".join(overlay))
            print(f"Preview written to {preview_path}")
            cmd = input("Command [space/a/d/j/k/n/e/s/q]: ").strip().lower()
            key = ord(cmd[:1]) if cmd else -1
            if cmd == "":
                key = ord("d")
        else:
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            h, w = canvas_bgr.shape[:2]
            if w > args.max_width:
                scale = args.max_width / w
                canvas_bgr = cv2.resize(canvas_bgr, (int(w * scale), int(h * scale)))
            cv2.imshow("libero-annotator", canvas_bgr)
            key = cv2.waitKey(delay if playing else 0)

        if playing:
            current = min(total_frames - 1, current + 1)
            if current == total_frames - 1:
                playing = False
            if key == ord(" "):
                playing = False
            elif key in (ord("q"), 27):
                break
            elif key == ord("s"):
                pass
            continue

        if key == ord("q") or key == 27:
            break
        if key == ord(" "):
            playing = not playing
            continue
        if key == ord("a"):
            current = max(0, current - args.frame_step)
        elif key == ord("d"):
            current = min(total_frames - 1, current + args.frame_step)
        elif key == ord("j"):
            current = max(0, current - 10)
        elif key == ord("k"):
            current = min(total_frames - 1, current + 10)
        elif key == ord("n"):
            if current < current_segment_start:
                print("Current frame is before segment start; ignored.")
                continue
            effect = choose_from_list("Effect", DEFAULT_EFFECTS, default=current_effect)
            stage = choose_from_list("Stage", DEFAULT_STAGES, default=current_stage)
            target = prompt_text("Target", default=current_target)
            segments.append(
                Segment(
                    start_frame=current_segment_start,
                    end_frame=current,
                    effect=effect,
                    stage=stage,
                    target=target,
                )
            )
            print(f"Saved segment: {current_segment_start} -> {current} | {effect} / {stage} / {target}")
            current_segment_start = current
            current_effect = effect
            current_stage = stage
            current_target = target
        elif key == ord("e"):
            current_effect = choose_from_list("Effect", DEFAULT_EFFECTS, default=current_effect)
            current_stage = choose_from_list("Stage", DEFAULT_STAGES, default=current_stage)
            current_target = prompt_text("Target", default=current_target)
        elif key == ord("s"):
            break

    payload = {
        "data_root": str(data_root),
        "episode_index": args.episode_index,
        "task": ep.get("tasks", []),
        "video_path": str(video_path),
        "fps": fps,
        "segments": [asdict(s) for s in segments],
        "partial_segment": {
            "start_frame": current_segment_start,
            "end_frame": current,
            "effect": current_effect,
            "stage": current_stage,
            "target": current_target,
        },
    }
    save_annotations(Path(args.output), payload)
    print(f"Saved annotations to {args.output}")
    if not headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
