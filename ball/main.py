import queue
import threading
import time
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# --- Configuración ---
MODEL_PATH = "G:/Mi unidad/ScoutingData/modelos/Yolo/ball/ball_v1/weights/best.pt"
VIDEO_INPUT = "G:/Mi unidad/ScoutingData/videosPartidos/ccVSutc_liga1max_2026_clausura_st.mp4"
VIDEO_OUTPUT = "G:/Mi unidad/utp clases/2026-3/Aquitectura Orientada al Servicio/Semana 15/proyectoSOA/scouringData_VideoInference/ball/video_out.mp4"

BALL_CLASS_ID = 0         # Ajusta al ID de tu dataset (en COCO es 32)
CONF_NORMAL = 0.25        # Confianza para la pasada normal
CONF_RESCUE = 0.20        # Confianza para ROI y SAHI
IMGSZ_NORMAL = 1280       # Resolución de tu modelo
ROI_SIZE = 640            # Tamaño de la ventana local alrededor del último balón
MAX_MISSED_FOR_ROI = 15   # Si pasan más de 15 frames sin balón, el ROI viejo se descarta
MAX_FRAMES = 100          # Límite de frames a procesar (None para todo el video)
LOG_INTERVAL = 1          # Mostrar progreso en consola cada N frames

# Parámetros de predicción de velocidad
MAX_BALL_SPEED = 150.0    # Máx. píxeles/frame válidos (evita saltos por cortes de cámara)
VEL_ALPHA = 0.7           # Peso de la nueva velocidad (EMA para reducir jitter)
VEL_DAMPING = 0.90        # Frenado progresivo (10% menos velocidad por cada frame perdido)

model = YOLO(MODEL_PATH)


def sahi_ball_callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(
        image_slice,
        conf=CONF_RESCUE,
        classes=[BALL_CLASS_ID],
        verbose=False
    )[0]
    return sv.Detections.from_ultralytics(result)


slicer = sv.InferenceSlicer(
    callback=sahi_ball_callback,
    slice_wh=(640, 640),
    overlap_wh=(128, 128),  # 20% de 640px en supervision 0.28+
    iou_threshold=0.4
)


def get_best_detection(det: sv.Detections) -> sv.Detections:
    """Devuelve solo la detección con mayor confianza."""
    if len(det) <= 1:
        return det
    best_idx = np.argmax(det.confidence)
    return det[best_idx : best_idx + 1]


def get_center(det: sv.Detections) -> tuple[float, float]:
    """Calcula el centro (cx, cy) de la primera caja."""
    x1, y1, x2, y2 = det.xyxy[0]
    return float((x1 + x2) / 2), float((y1 + y2) / 2)


def get_roi_crop(frame: np.ndarray, center: tuple[float, float], size: int):
    """Extrae un recorte cuadrado respetando los bordes del frame."""
    h, w = frame.shape[:2]
    cx = int(np.clip(center[0], 0, w))
    cy = int(np.clip(center[1], 0, h))

    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)

    # Reajustar origen si el recorte tocó el borde derecho o inferior
    x1 = max(0, x2 - size)
    y1 = max(0, y2 - size)

    return frame[y1:y2, x1:x2], (x1, y1)


def update_velocity(
    prev_center: tuple[float, float] | None,
    curr_center: tuple[float, float],
    prev_vel: tuple[float, float],
    dt: int
) -> tuple[float, float]:
    """Actualiza (vx, vy) con suavizado EMA y descarta saltos imposibles."""
    if prev_center is None or dt <= 0:
        return (0.0, 0.0)

    raw_vx = (curr_center[0] - prev_center[0]) / dt
    raw_vy = (curr_center[1] - prev_center[1]) / dt
    speed = np.hypot(raw_vx, raw_vy)

    # Si el salto supera el límite físico (ej. cambio de plano), reinicia la velocidad
    if speed > MAX_BALL_SPEED:
        return (0.0, 0.0)

    vx = VEL_ALPHA * raw_vx + (1.0 - VEL_ALPHA) * prev_vel[0]
    vy = VEL_ALPHA * raw_vy + (1.0 - VEL_ALPHA) * prev_vel[1]
    return (vx, vy)


class ThreadedOpenCVReader:
    """Lee frames de video en un hilo secundario usando una cola acotada."""

    def __init__(self, video_path: str, queue_size: int = 64) -> None:
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")
        self.queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            while not self._stop_event.is_set():
                try:
                    self.queue.put(frame, timeout=0.1)
                    break
                except queue.Full:
                    continue
        # Señal de fin (sin bloquear si ya se pidió detener)
        try:
            self.queue.put(None, timeout=0.1)
        except queue.Full:
            pass

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        frame = self.queue.get()
        if frame is None:
            raise StopIteration
        return frame

    def stop(self) -> None:
        self._stop_event.set()
        # Vaciar un elemento si el hilo está bloqueado en put()
        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


# --- Estado y métricas ---
box_annotator = sv.BoxAnnotator()
video_info = sv.VideoInfo.from_video_path(VIDEO_INPUT)

total_frames = video_info.total_frames
if MAX_FRAMES is not None:
    total_frames = min(total_frames, MAX_FRAMES) if total_frames else MAX_FRAMES

last_ball_center = None
pred_center = None
ball_velocity = (0.0, 0.0)
missed_frames = 0
processed_frames = 0
stats = {"normal": 0, "roi": 0, "sahi": 0, "lost": 0}
start_time = time.perf_counter()

# --- Bucle principal ---
with ThreadedOpenCVReader(VIDEO_INPUT) as reader, sv.VideoSink(VIDEO_OUTPUT, video_info=video_info) as sink:
    for frame_idx, frame in enumerate(reader):
        if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
            break

        # --- NIVEL 1: Inferencia normal ---
        result = model(frame, imgsz=IMGSZ_NORMAL, conf=CONF_NORMAL, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        ball_det = detections[detections.class_id == BALL_CLASS_ID]

        if len(ball_det) > 0:
            best_ball = get_best_detection(ball_det)
            curr_center = get_center(best_ball)

            ball_velocity = update_velocity(
                last_ball_center, curr_center, ball_velocity, dt=missed_frames + 1
            )
            last_ball_center = curr_center
            pred_center = curr_center
            missed_frames = 0
            stats["normal"] += 1
        else:
            rescued_ball = sv.Detections.empty()

            # --- NIVEL 2: ROI local con predicción lineal ---
            if pred_center is not None and missed_frames < MAX_MISSED_FOR_ROI:
                # Avanzar el centro predicho aplicando amortiguación
                vx, vy = ball_velocity[0] * VEL_DAMPING, ball_velocity[1] * VEL_DAMPING
                ball_velocity = (vx, vy)
                pred_center = (pred_center[0] + vx, pred_center[1] + vy)

                crop, (ox, oy) = get_roi_crop(frame, pred_center, ROI_SIZE)
                roi_result = model(
                    crop,
                    imgsz=ROI_SIZE,
                    conf=CONF_RESCUE,
                    classes=[BALL_CLASS_ID],
                    verbose=False
                )[0]
                roi_det = sv.Detections.from_ultralytics(roi_result)

                if len(roi_det) > 0:
                    rescued_ball = get_best_detection(roi_det)
                    rescued_ball.xyxy += np.array([ox, oy, ox, oy], dtype=np.float32)
                    stats["roi"] += 1

            # --- NIVEL 3: Fallback a SAHI completo ---
            if len(rescued_ball) == 0:
                sahi_det = slicer(frame)
                if len(sahi_det) > 0:
                    rescued_ball = get_best_detection(sahi_det)
                    stats["sahi"] += 1

            # Actualizar estado tras el rescate
            if len(rescued_ball) > 0:
                curr_center = get_center(rescued_ball)
                ball_velocity = update_velocity(
                    last_ball_center, curr_center, ball_velocity, dt=missed_frames + 1
                )
                last_ball_center = curr_center
                pred_center = curr_center
                missed_frames = 0
                detections = sv.Detections.merge([detections, rescued_ball])
            else:
                missed_frames += 1
                stats["lost"] += 1

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        sink.write_frame(annotated_frame)

        processed_frames += 1
        if processed_frames % LOG_INTERVAL == 0 or (total_frames and processed_frames == total_frames):
            elapsed = time.perf_counter() - start_time
            fps = processed_frames / elapsed if elapsed > 0 else 0.0
            if total_frames:
                pct = (processed_frames / total_frames) * 100
                eta_sec = (total_frames - processed_frames) / fps if fps > 0 else 0.0
                print(
                    f"Frame {processed_frames}/{total_frames} ({pct:.1f}%) "
                    f"| {fps:.2f} FPS | ETA: {eta_sec:.0f}s | {stats}",
                    flush=True,
                )
            else:
                print(
                    f"Frame {processed_frames} | {fps:.2f} FPS | {stats}",
                    flush=True,
                )

print(f"Proceso terminado. Frames procesados: {processed_frames}. Resumen -> {stats}")
