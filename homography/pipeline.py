"""Pipeline standalone de homografía para procesar un video con el modelo .pt."""

import argparse
import json
from pathlib import Path
from homography.infer import run_homography_inference

DEFAULT_VIDEO_PATH = "G:/Mi unidad/ScoutingData/videosPartidos/ccVSutc_liga1max_2026_clausura_st.mp4"
DEFAULT_MODEL_PATH = r"G:\Mi unidad\ScoutingData\modelos\Yolo\homography\homography_v3\weights\best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de homografía: detecta keypoints de cancha, calcula posiciones reales y distancia en metros."
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Ruta al video de entrada (.mp4)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Ruta al modelo YOLO pose (.pt)")
    parser.add_argument("--output-dir", default="outputs/homography", help="Directorio de salida para CSV, video y métricas")
    parser.add_argument("--imgsz", type=int, default=1280, help="Resolución de inferencia")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza de detección de cancha")
    parser.add_argument("--kpt-conf", type=float, default=0.35, help="Umbral mínimo de confianza por keypoint")
    parser.add_argument("--ransac-thresh-m", type=float, default=2.5, help="Umbral RANSAC en metros")
    parser.add_argument("--max-frames", type=int, default=1000, help="Límite de frames a procesar (por defecto 5000; -1 para todo el video)")
    parser.add_argument("--stride", type=int, default=1, help="Salto de frames (1 = todos)")
    parser.add_argument("--device", default="0", help="Dispositivo ('0' para GPU CUDA o 'cpu')")
    parser.add_argument("--no-wireframe", action="store_true", help="Desactiva el wireframe reproyectado")
    parser.add_argument("--no-minimap", action="store_true", help="Desactiva el minimapa 2D en esquina")
    parser.add_argument("--no-stabilize", action="store_true", help="Desactiva la estabilización temporal de keypoints y H")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device: int | str = int(args.device) if str(args.device).isdigit() else args.device
    max_frames = None if (args.max_frames is not None and args.max_frames < 0) else args.max_frames

    cfg = {
        "name": "pt",
        "model_format": "pt",
        "device": device,
        "video": args.video,
        "model": args.model,
        "task_output_dir": str(out_dir),
        "videos_output_dir": str(out_dir),
        "inference": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "max_frames": max_frames,
            "frame_stride": args.stride,
            "log_interval": 20,
        },
        "homography": {
            "conf": args.conf,
            "kpt_conf": args.kpt_conf,
            "ransac_thresh_m": args.ransac_thresh_m,
            "stabilize": not args.no_stabilize,
            "write_video": True,
            "draw_wireframe": not args.no_wireframe,
            "draw_minimap": not args.no_minimap,
        },
    }

    metrics = run_homography_inference(cfg)
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, mode="w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n=== Pipeline de homografía finalizado. Métricas en {metrics_path} ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
