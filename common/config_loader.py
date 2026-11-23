from pathlib import Path
from typing import Any
import yaml


def load_config(
    config_path: str,
    task: str,
    video_override: str | None = None,
    model_override: str | None = None,
    output_dir_override: str | None = None,
    max_frames_override: int | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Carga un YAML de configuración y aplica overrides recibidos por CLI."""
    with open(config_path, mode="r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if video_override is not None:
        cfg["video"] = video_override

    models = cfg.setdefault("models", {})
    if model_override is not None:
        models[task] = model_override
    cfg["model"] = models.get(task)

    if not cfg.get("model"):
        raise ValueError(f"No se definió ruta de modelo para la tarea '{task}' ni se pasó --model.")

    base_out = Path(output_dir_override or cfg.get("output_dir", "outputs"))
    task_out = base_out / task
    task_out.mkdir(parents=True, exist_ok=True)
    cfg["task_output_dir"] = str(task_out)

    videos_out = Path(cfg.get("videos_dir", "outputs/videos"))
    videos_out.mkdir(parents=True, exist_ok=True)
    cfg["videos_output_dir"] = str(videos_out)

    inf = cfg.setdefault("inference", {})
    if max_frames_override is not None:
        inf["max_frames"] = None if max_frames_override < 0 else max_frames_override
    if device_override is not None:
        cfg["device"] = int(device_override) if device_override.isdigit() else device_override

    return cfg
