import csv
import time
from pathlib import Path
from typing import Any
import supervision as sv
from ultralytics import YOLO
from common.video_reader import ThreadedOpenCVReader


def run_player_inference(cfg: dict[str, Any]) -> dict[str, Any]:
    """Ejecuta detección de jugadores según YAML y guarda CSV + métricas."""
    video_path = cfg["video"]
    model_path = cfg["model"]
    device = cfg.get("device", 0)
    out_dir = Path(cfg["task_output_dir"])
    inf_cfg = cfg["inference"]
    player_cfg = cfg.get("player", {})

    classes = player_cfg.get("classes")
    imgsz = int(inf_cfg.get("imgsz", 1280))
    conf_thresh = float(inf_cfg.get("conf", 0.25))
    max_frames = inf_cfg.get("max_frames")
    stride = int(inf_cfg.get("frame_stride", 1))
    write_video = bool(inf_cfg.get("write_raw_video", False))
    log_interval = int(inf_cfg.get("log_interval", 20))

    csv_output = out_dir / "detections_player.csv"
    profile_name = cfg.get("name", "run")
    video_output = Path(cfg.get("videos_output_dir", out_dir)) / f"player_{profile_name}.mp4"

    model = YOLO(model_path, task="detect")
    video_info = sv.VideoInfo.from_video_path(video_path)
    total_frames = min(video_info.total_frames, max_frames) if (max_frames and video_info.total_frames) else (max_frames or video_info.total_frames)

    box_annotator = sv.BoxAnnotator() if write_video else None
    sink = sv.VideoSink(str(video_output), video_info=video_info) if write_video else None
    if sink is not None:
        sink.__enter__()

    processed_frames = 0
    total_detections = 0
    start_time = time.perf_counter()

    try:
        with ThreadedOpenCVReader(video_path, max_frames=max_frames, stride=stride) as reader, open(
            csv_output, mode="w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["frame", "class_id", "confidence", "x1", "y1", "x2", "y2", "cx", "cy", "width", "height"])

            for frame_idx, frame in reader:
                result = model(
                    frame,
                    imgsz=imgsz,
                    conf=conf_thresh,
                    classes=classes,
                    device=device,
                    verbose=False,
                )[0]
                detections = sv.Detections.from_ultralytics(result)

                for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
                    cls_id = int(detections.class_id[i]) if detections.class_id is not None else -1
                    conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                    writer.writerow([
                        frame_idx, cls_id, f"{conf:.4f}",
                        f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                        f"{(x1 + x2) / 2.0:.2f}", f"{(y1 + y2) / 2.0:.2f}",
                        f"{x2 - x1:.2f}", f"{y2 - y1:.2f}",
                    ])
                    total_detections += 1

                if sink is not None and box_annotator is not None:
                    sink.write_frame(box_annotator.annotate(scene=frame.copy(), detections=detections))

                processed_frames += 1
                if processed_frames % log_interval == 0 or (total_frames and frame_idx + 1 >= total_frames):
                    elapsed = time.perf_counter() - start_time
                    fps = processed_frames / elapsed if elapsed > 0 else 0.0
                    print(f"[PLAYER] Frame {frame_idx + 1}/{total_frames} | {fps:.2f} FPS | Detecciones: {total_detections}", flush=True)
    finally:
        if sink is not None:
            sink.__exit__(None, None, None)

    elapsed_total = time.perf_counter() - start_time
    return {
        "csv_player": str(csv_output),
        "processed_frames": processed_frames,
        "total_detections": total_detections,
        "elapsed_s": round(elapsed_total, 2),
        "fps": round(processed_frames / elapsed_total, 2) if elapsed_total > 0 else 0.0,
    }
