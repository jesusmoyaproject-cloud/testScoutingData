import csv
import json
import time
from pathlib import Path
from typing import Any
import supervision as sv
from ultralytics import YOLO
from common.video_reader import ThreadedOpenCVReader


def run_homography_inference(cfg: dict[str, Any]) -> dict[str, Any]:
    """Ejecuta detección de keypoints de cancha (pose) según YAML y guarda CSV + métricas."""
    video_path = cfg["video"]
    model_path = cfg["model"]
    device = cfg.get("device", 0)
    out_dir = Path(cfg["task_output_dir"])
    inf_cfg = cfg["inference"]
    homo_cfg = cfg.get("homography", {})

    imgsz = int(inf_cfg.get("imgsz", 1280))
    conf_thresh = float(homo_cfg.get("conf", inf_cfg.get("conf", 0.25)))
    max_frames = inf_cfg.get("max_frames")
    stride = int(inf_cfg.get("frame_stride", 1))
    log_interval = int(inf_cfg.get("log_interval", 20))

    csv_output = out_dir / "keypoints_homography.csv"
    model = YOLO(model_path, task="pose")
    video_info = sv.VideoInfo.from_video_path(video_path)
    total_frames = min(video_info.total_frames, max_frames) if (max_frames and video_info.total_frames) else (max_frames or video_info.total_frames)

    processed_frames = 0
    frames_with_kpts = 0
    start_time = time.perf_counter()

    with ThreadedOpenCVReader(video_path, max_frames=max_frames, stride=stride) as reader, open(
        csv_output, mode="w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "num_keypoints", "keypoints_xy_conf"])

        for frame_idx, frame in reader:
            result = model(
                frame,
                imgsz=imgsz,
                conf=conf_thresh,
                device=device,
                verbose=False,
            )[0]

            kpts_list = []
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                kpts_data = result.keypoints.data[0].cpu().numpy()
                kpts_list = [[round(float(x), 2), round(float(y), 2), round(float(c), 4)] for x, y, c in kpts_data]
                frames_with_kpts += 1

            writer.writerow([frame_idx, len(kpts_list), json.dumps(kpts_list)])
            processed_frames += 1

            if processed_frames % log_interval == 0 or (total_frames and frame_idx + 1 >= total_frames):
                elapsed = time.perf_counter() - start_time
                fps = processed_frames / elapsed if elapsed > 0 else 0.0
                print(f"[HOMOGRAPHY] Frame {frame_idx + 1}/{total_frames} | {fps:.2f} FPS", flush=True)

    elapsed_total = time.perf_counter() - start_time
    return {
        "csv_homography": str(csv_output),
        "processed_frames": processed_frames,
        "frames_with_keypoints": frames_with_kpts,
        "elapsed_s": round(elapsed_total, 2),
        "fps": round(processed_frames / elapsed_total, 2) if elapsed_total > 0 else 0.0,
    }
