import argparse
import json
from pathlib import Path
from common.config_loader import load_config
from ball.infer import run_ball_inference
from ball.postprocess import run_ball_postprocess
from player.infer import run_player_inference
from player.postprocess import run_player_postprocess
from homography.infer import run_homography_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ScoutingData Video Inference Pipeline")
    parser.add_argument("--config", required=True, help="Ruta al archivo .yaml (openvino.yaml, pt.yaml, engine.yaml)")
    parser.add_argument("--task", required=True, choices=["ball", "player", "homography"], help="Módulo a ejecutar")
    parser.add_argument("--video", default=None, help="Sobreescribe la ruta del video definida en el YAML")
    parser.add_argument("--model", default=None, help="Sobreescribe la ruta del modelo definida en el YAML")
    parser.add_argument("--output-dir", default=None, help="Sobreescribe el directorio de salida")
    parser.add_argument("--max-frames", type=int, default=None, help="Límite de frames (-1 para procesar todo el video)")
    parser.add_argument("--device", default=None, help="Sobreescribe device (ej. 0, cpu, intel:gpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(
        config_path=args.config,
        task=args.task,
        video_override=args.video,
        model_override=args.model,
        output_dir_override=args.output_dir,
        max_frames_override=args.max_frames,
        device_override=args.device,
    )

    print(f"=== Ejecutando task='{args.task}' | profile='{cfg.get('name')}' | device='{cfg.get('device')}' ===")
    print(f"Video: {cfg['video']}")
    print(f"Model: {cfg['model']}")

    if args.task == "ball":
        metrics = run_ball_inference(cfg)
        if cfg.get("ball", {}).get("run_postprocess", True):
            pp_metrics = run_ball_postprocess(cfg, Path(metrics["csv_raw"]))
            metrics["postprocess"] = pp_metrics
    elif args.task == "player":
        metrics = run_player_inference(cfg)
        if cfg.get("player", {}).get("run_postprocess", True):
            pp_metrics = run_player_postprocess(cfg, Path(metrics["csv_player"]))
            metrics["postprocess"] = pp_metrics
    else:
        metrics = run_homography_inference(cfg)

    metrics["task"] = args.task
    metrics["profile"] = cfg.get("name")
    metrics["model_format"] = cfg.get("model_format")
    metrics["device"] = cfg.get("device")
    metrics["video"] = cfg["video"]
    metrics["model"] = cfg["model"]

    metrics_path = Path(cfg["task_output_dir"]) / "metrics.json"
    with open(metrics_path, mode="w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n=== Métricas guardadas en {metrics_path} ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
