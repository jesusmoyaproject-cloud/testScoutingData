# ScoutingData Video Inference

Pipeline modular para inferencia y benchmarking de **balón**, **jugadores** y **homografía** en videos de fútbol, parametrizado con perfiles YAML para **OpenVINO**, **PyTorch (`.pt`)** y **TensorRT (`.engine`)**.

## Estructura del proyecto

```text
scouringData_VideoInference/
├── configs/
│   ├── openvino.yaml      # Perfil local Intel (CPU / iGPU / NPU)
│   ├── pt.yaml            # Perfil PyTorch (.pt)
│   └── engine.yaml        # Perfil TensorRT (.engine) para Colab L4/T4
├── common/
│   ├── config_loader.py   # Carga YAML + overrides por CLI
│   └── video_reader.py    # ThreadedOpenCVReader compartido
├── ball/
│   ├── infer.py           # Inferencia de balón -> detections_raw.csv
│   └── postprocess.py     # Kalman + ByteTrack + Viterbi DP + PCHIP -> detections_clean.csv
├── player/
│   └── infer.py           # Inferencia de jugadores -> detections_player.csv
├── homography/
│   └── infer.py           # Inferencia de keypoints -> keypoints_homography.csv
└── run_pipeline.py        # Punto de entrada único con métricas en metrics.json
```

## Uso en local (OpenVINO)

```bash
pip install -r requirements.txt

python run_pipeline.py \
  --task ball \
  --config configs/openvino.yaml \
  --video "G:/Mi unidad/ScoutingData/videosPartidos/ccVSutc_liga1max_2026_clausura_st.mp4" \
  --model "G:/Mi unidad/ScoutingData/modelos/OpenVino/yolov11m-ball_openvino_model"
```

## Uso en Google Colab (L4 + `.engine`)

```bash
# 1. Clonar repositorio e instalar dependencias
!git clone <URL_DE_TU_REPO>
%cd scouringData_VideoInference
!pip install -r requirements.txt

# 2. Copiar video y modelo al disco local de Colab (/content/) para máximo I/O
# (Opcional) Exportar .pt a .engine directamente en la L4:
!yolo export model=/content/best.pt format=engine imgsz=1280 half=True

# 3. Ejecutar pipeline de balón (u otra tarea: player / homography)
!python run_pipeline.py \
  --task ball \
  --config configs/engine.yaml \
  --video /content/partido.mp4 \
  --model /content/best.engine \
  --output-dir /content/outputs/engine \
  --max-frames -1
```

## Salidas generadas por ejecución
Cada corrida guarda en `<output_dir>/<task>/`:
- `metrics.json`: FPS, tiempo total, frames procesados, cobertura y resumen de post-procesamiento para comparar `.pt` vs `openvino` vs `.engine`.
- `detections_raw.csv` y `detections_clean.csv` (en `--task ball`).
- `detections_player.csv` (en `--task player`).
- `keypoints_homography.csv` (en `--task homography`).
