"""Inferencia de homografía con EMA doble pasado (forward+backward), keypoint real, CSV y video."""

import csv
import time
from pathlib import Path
from typing import Any
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO
from common.video_reader import ThreadedOpenCVReader
from homography.geometry import (
    compute_distances_m,
    compute_homography,
    project_points,
    reproject_real_to_px,
)
from homography.pitch_config import (
    SoccerPitchConfiguration,
    build_edges_for_model,
    build_keypoint_dictionary,
)
from homography.stabilizer import HomographyStabilizer
from homography.visualizer import render_homography_frame


def _resolve_device(requested_device: Any) -> Any:
    """Resuelve el dispositivo de inferencia haciendo fallback a CPU si no hay CUDA."""
    if requested_device == "cpu":
        return "cpu"
    if isinstance(requested_device, int) or str(requested_device).startswith("cuda") or str(requested_device).isdigit():
        if not torch.cuda.is_available():
            print(f"[HOMOGRAPHY] CUDA no disponible; usando 'cpu' en lugar de '{requested_device}'.", flush=True)
            return "cpu"
    return requested_device


def run_homography_inference(cfg: dict[str, Any]) -> dict[str, Any]:
    """Ejecuta inferencia (pasada 1), EMA doble pasado (pasada 2), CSV y video anotado."""
    video_path = cfg["video"]
    model_path = cfg["model"]
    device = _resolve_device(cfg.get("device", 0))
    out_dir = Path(cfg["task_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    inf_cfg = cfg.get("inference", {})
    homo_cfg = cfg.get("homography", {})

    imgsz = int(inf_cfg.get("imgsz", 1280))
    conf_thresh = float(homo_cfg.get("conf", inf_cfg.get("conf", 0.25)))
    kpt_conf_thresh = float(homo_cfg.get("kpt_conf", 0.35))
    ransac_thresh_m = float(homo_cfg.get("ransac_thresh_m", 2.0))
    stabilize = bool(homo_cfg.get("stabilize", True))
    max_jump_px = float(homo_cfg.get("max_jump_px", 35.0))
    max_geom_err_m = float(homo_cfg.get("max_geom_err_m", 2.5))
    max_hold_frames = int(homo_cfg.get("max_hold_frames", 2))
    kpt_ema_alpha = float(homo_cfg.get("kpt_ema_alpha", 0.45))
    h_ema_alpha = float(homo_cfg.get("h_ema_alpha", 0.45))
    max_frames = inf_cfg.get("max_frames")
    stride = int(inf_cfg.get("frame_stride", 1))
    log_interval = int(inf_cfg.get("log_interval", 20))
    write_video = bool(homo_cfg.get("write_video", True))
    draw_wireframe = bool(homo_cfg.get("draw_wireframe", True))
    draw_minimap = bool(homo_cfg.get("draw_minimap", True))

    csv_output = out_dir / "keypoints_homography.csv"
    profile_name = cfg.get("name", "homography")
    videos_dir = Path(cfg.get("videos_output_dir", out_dir))
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_output = Path(homo_cfg.get("video_output") or (videos_dir / f"homography_{profile_name}.mp4"))

    model = YOLO(model_path, task="pose")
    kpt_shape = getattr(model.model, "kpt_shape", None)
    num_model_kpts = int(kpt_shape[0]) if kpt_shape else 30

    pitch_cfg = SoccerPitchConfiguration()
    specs = build_keypoint_dictionary(num_keypoints=num_model_kpts, config=pitch_cfg)
    specs_by_idx = {s.model_idx: s for s in specs}
    edges = build_edges_for_model(specs, config=pitch_cfg)

    stabilizer = (
        HomographyStabilizer(
            max_jump_px=max_jump_px,
            max_geom_err_m=max_geom_err_m,
            max_hold_frames=max_hold_frames,
            h_ema_alpha=h_ema_alpha,
            kpt_ema_alpha=kpt_ema_alpha,
        )
        if stabilize
        else None
    )

    video_info = sv.VideoInfo.from_video_path(video_path)
    total_frames = video_info.total_frames
    if max_frames is not None:
        total_frames = min(total_frames, max_frames) if total_frames else max_frames

    frames_data: list[dict[str, Any]] = []
    total_jumps_rejected = 0
    start_time = time.perf_counter()

    # --- PASADA 1: Inferencia YOLO + detección de cortes + filtrado de saltos/derivas ---
    with ThreadedOpenCVReader(video_path, max_frames=max_frames, stride=stride) as reader:
        for frame_idx, frame in reader:
            scene_cut = stabilizer.detect_scene_cut(frame) if stabilizer is not None else False

            result = model(
                frame,
                imgsz=imgsz,
                conf=conf_thresh,
                device=device,
                verbose=False,
            )[0]

            raw_records: list[dict[str, Any]] = []
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                kpts_data = result.keypoints.data[0].cpu().numpy()
                for idx, (x_px, y_px, k_conf) in enumerate(kpts_data):
                    if idx >= len(specs) or float(k_conf) < kpt_conf_thresh:
                        continue
                    spec = specs[idx]
                    raw_records.append({
                        "keypoint_idx": spec.model_idx,
                        "roboflow_id": spec.roboflow_id,
                        "keypoint_name": spec.name,
                        "confidence": float(k_conf),
                        "det_x_px": float(x_px),
                        "det_y_px": float(y_px),
                        "raw_x_px": float(x_px),
                        "raw_y_px": float(y_px),
                        "real_x_m": spec.x_m,
                        "real_y_m": spec.y_m,
                        "proj_x_m": None,
                        "proj_y_m": None,
                        "real_x_px": None,
                        "real_y_px": None,
                        "dist_m": None,
                        "is_inlier": False,
                        "is_outlier": False,
                        "source": "detected",
                        "color_bgr": spec.color_bgr,
                    })

            if stabilizer is not None:
                active_records, outlier_records = stabilizer.update_keypoints(
                    frame_idx=frame_idx,
                    raw_records=raw_records,
                    specs_by_idx=specs_by_idx,
                )
                total_jumps_rejected += len(outlier_records)

                h_cand = None
                if len(active_records) >= 4:
                    src_px = np.array([[r["det_x_px"], r["det_y_px"]] for r in active_records], dtype=np.float32)
                    dst_m = np.array([[r["real_x_m"], r["real_y_m"]] for r in active_records], dtype=np.float32)
                    h_cand, _ = compute_homography(
                        src_px, dst_m, ransac_thresh_m=ransac_thresh_m, refine_inliers=True
                    )
                stabilizer.stabilize_homography(frame_idx, h_cand, frame.shape[:2])
                h_matrix = None
            else:
                active_records = raw_records
                outlier_records = []
                h_matrix = None
                if len(active_records) >= 4:
                    src_px = np.array([[r["det_x_px"], r["det_y_px"]] for r in active_records], dtype=np.float32)
                    dst_m = np.array([[r["real_x_m"], r["real_y_m"]] for r in active_records], dtype=np.float32)
                    h_matrix, _ = compute_homography(
                        src_px, dst_m, ransac_thresh_m=ransac_thresh_m, refine_inliers=True
                    )
                    if h_matrix is not None:
                        proj_m = project_points(h_matrix, src_px)
                        dists_m = compute_distances_m(proj_m, dst_m)
                        reproj_px = reproject_real_to_px(h_matrix, dst_m)
                        for i_r, (r, (px_m, py_m), d_m) in enumerate(zip(active_records, proj_m, dists_m)):
                            r["proj_x_m"] = float(px_m)
                            r["proj_y_m"] = float(py_m)
                            r["dist_m"] = float(d_m)
                            r["is_inlier"] = bool(d_m <= ransac_thresh_m)
                            r["is_outlier"] = not r["is_inlier"]
                            if reproj_px is not None:
                                r["real_x_px"] = float(reproj_px[i_r, 0])
                                r["real_y_px"] = float(reproj_px[i_r, 1])

            frames_data.append({
                "frame_idx": frame_idx,
                "scene_cut": scene_cut,
                "active_records": active_records,
                "outlier_records": outlier_records,
                "h_matrix": h_matrix,
            })

            processed = len(frames_data)
            if processed % log_interval == 0 or (total_frames and frame_idx + 1 >= total_frames):
                elapsed = time.perf_counter() - start_time
                fps = processed / elapsed if elapsed > 0 else 0.0
                pct = ((frame_idx + 1) / total_frames) * 100 if total_frames else 0.0
                print(
                    f"[HOMOGRAPHY Pasada 1/2] Frame {frame_idx + 1}/{total_frames} ({pct:.1f}%) "
                    f"| {fps:.2f} FPS | Kpts: {len(active_records)}",
                    flush=True,
                )

    # --- PASADA 2: EMA Doble Pasado (Forward + Backward) sin desfase ---
    if stabilizer is not None:
        print("[HOMOGRAPHY Pasada 2/2] Aplicando EMA doble pasado (forward + backward)...", flush=True)
        stabilizer.apply_double_pass_ema(
            frames_data=frames_data,
            specs_by_idx=specs_by_idx,
            ransac_thresh_m=ransac_thresh_m,
        )

    # --- Guardar CSV y estadísticas ---
    frames_with_kpts = 0
    frames_with_valid_h = 0
    total_kpts_detected = 0
    all_dists_m: list[float] = []

    with open(csv_output, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "frame",
            "keypoint_idx",
            "roboflow_id",
            "keypoint_name",
            "confidence",
            "det_x_px",
            "det_y_px",
            "real_x_m",
            "real_y_m",
            "proj_x_m",
            "proj_y_m",
            "dist_m",
            "homography_valid",
            "num_kpts_frame",
            "raw_x_px",
            "raw_y_px",
            "real_x_px",
            "real_y_px",
            "is_inlier",
            "is_outlier",
            "source",
        ])

        for fd in frames_data:
            frame_idx = fd["frame_idx"]
            active_records = fd["active_records"]
            outlier_records = fd["outlier_records"]
            h_matrix = fd["h_matrix"]
            h_valid = h_matrix is not None

            num_kpts_frame = len(active_records)
            if num_kpts_frame > 0:
                frames_with_kpts += 1
                total_kpts_detected += num_kpts_frame
            if h_valid:
                frames_with_valid_h += 1

            for r in active_records + outlier_records:
                if r.get("dist_m") is not None and r.get("is_inlier", False):
                    all_dists_m.append(float(r["dist_m"]))
                writer.writerow([
                    frame_idx,
                    r["keypoint_idx"],
                    r["roboflow_id"],
                    r["keypoint_name"],
                    f"{r['confidence']:.4f}",
                    f"{r['det_x_px']:.2f}",
                    f"{r['det_y_px']:.2f}",
                    f"{r['real_x_m']:.2f}",
                    f"{r['real_y_m']:.2f}",
                    f"{r['proj_x_m']:.2f}" if r["proj_x_m"] is not None else "",
                    f"{r['proj_y_m']:.2f}" if r["proj_y_m"] is not None else "",
                    f"{r['dist_m']:.4f}" if r["dist_m"] is not None else "",
                    h_valid,
                    num_kpts_frame,
                    f"{r['raw_x_px']:.2f}",
                    f"{r['raw_y_px']:.2f}",
                    f"{r['real_x_px']:.2f}" if r.get("real_x_px") is not None else "",
                    f"{r['real_y_px']:.2f}" if r.get("real_y_px") is not None else "",
                    r.get("is_inlier", False),
                    r.get("is_outlier", False),
                    r.get("source", "detected"),
                ])

    # --- Renderizado final del video ---
    if write_video:
        print(f"[HOMOGRAPHY] Renderizando video final en {video_output}...", flush=True)
        by_frame_idx = {fd["frame_idx"]: fd for fd in frames_data}
        with ThreadedOpenCVReader(video_path, max_frames=max_frames, stride=stride) as reader, sv.VideoSink(
            str(video_output), video_info=video_info
        ) as sink:
            for frame_idx, frame in reader:
                fd = by_frame_idx.get(frame_idx)
                if fd is None:
                    sink.write_frame(frame)
                    continue
                active_records = fd["active_records"]
                outlier_records = fd["outlier_records"]
                h_matrix = fd["h_matrix"]
                render_records = (
                    [r for r in active_records if r.get("is_inlier", False)]
                    if h_matrix is not None
                    else active_records
                )
                annotated = render_homography_frame(
                    frame=frame,
                    frame_idx=frame_idx,
                    detected_records=render_records,
                    homography_matrix=h_matrix,
                    pitch_cfg=pitch_cfg,
                    specs=specs,
                    edges=edges,
                    draw_wireframe=draw_wireframe,
                    draw_minimap=draw_minimap,
                    rejected_count=len(outlier_records) + (len(active_records) - len(render_records)),
                )
                sink.write_frame(annotated)

    elapsed_total = time.perf_counter() - start_time
    processed_frames = len(frames_data)
    return {
        "csv_homography": str(csv_output),
        "video_homography": str(video_output) if write_video else None,
        "processed_frames": processed_frames,
        "frames_with_keypoints": frames_with_kpts,
        "frames_with_valid_homography": frames_with_valid_h,
        "total_keypoints_detected": total_kpts_detected,
        "total_jumps_rejected": total_jumps_rejected,
        "mean_dist_m": round(float(np.mean(all_dists_m)), 4) if all_dists_m else None,
        "median_dist_m": round(float(np.median(all_dists_m)), 4) if all_dists_m else None,
        "max_dist_m": round(float(np.max(all_dists_m)), 4) if all_dists_m else None,
        "elapsed_s": round(elapsed_total, 2),
        "fps": round(processed_frames / elapsed_total, 2) if elapsed_total > 0 else 0.0,
    }
