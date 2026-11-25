import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm

# Rutas de archivos (usa el prefijo 'r' para evitar errores con las barras invertidas de Windows)
MODEL_PATH = r"G:\Mi unidad\ScoutingData\modelos\OpenVino\yolov11m-player_openvino_model"
SOURCE_VIDEO_PATH = r"G:\Mi unidad\ScoutingData\videosPartidos\ccVSgrau_liga1max_2026_clausura_pt.mp4"
TARGET_VIDEO_PATH = r"G:\Mi unidad\utp clases\2026-3\Aquitectura Orientada al Servicio\Semana 15\proyectoSOA\scouringData_VideoInference\outputs\videos\video_out_player.mp4"

# Parámetros de configuración
CONFIDENCE_THRESHOLD = 0.5
MAX_FRAMES = 3000

# Cargar el modelo OpenVINO
model = YOLO(MODEL_PATH, task="detect")

# Obtener metadatos del video original
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
total_frames = min(video_info.total_frames, MAX_FRAMES)

# Configurar anotadores de Supervision
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Generador de frames acotado a los primeros 3000 frames
frames_generator = sv.get_video_frames_generator(
    source_path=SOURCE_VIDEO_PATH,
    end=MAX_FRAMES
)

# Leer, inferir, anotar y guardar el nuevo video
with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info) as sink:
    for frame in tqdm(frames_generator, total=total_frames, desc="Procesando video"):
        # Inferencia con umbral de confianza de 0.5
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Generar etiquetas con nombre de clase y confianza
        labels = [
            f"{model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Dibujar cajas y etiquetas sobre el frame
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        # Escribir el frame en el video de salida
        sink.write_frame(frame=annotated_frame)

print(f"Video generado exitosamente en: {TARGET_VIDEO_PATH}")