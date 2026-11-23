import csv
import queue
import threading
import time
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# --- Configuración ---
MODEL_PATH = "G:/Mi unidad/ScoutingData/modelos/OpenVino/yolov11m-ball_openvino_model"
VIDEO_INPUT = "G:/Mi unidad/ScoutingData/videosPartidos/ccVSutc_liga1max_2026_clausura_st.mp4"
VIDEO_OUTPUT = "G:/Mi unidad/utp clases/2026-3/Aquitectura Orientada al Servicio/Semana 15/proyectoSOA/scouringData_VideoInference/ball/video_out_baseline.mp4"
CSV_OUTPUT = "G:/Mi unidad/utp clases/2026-3/Aquitectura Orientada al Servicio/Semana 15/proyectoSOA/scouringData_VideoInference/ball/detections_baseline.csv"

DEVICE = "intel:gpu"    # OpenVINO detectó ['CPU', 'GPU', 'NPU']; "intel:gpu" acelera sin perder precisión
BALL_CLASS_ID = 0       # Ajusta al ID de tu dataset (en COCO es 32)
CONF_NORMAL = 0.25      # Misma confianza base que main.py para comparar
IMGSZ_NORMAL = 1280     # Misma resolución que main.py
MAX_FRAMES = 5000       # Mismo límite de frames que main.py (None para todo el video)
FRAME_STRIDE = 1        # 1 = todos los frames; 2 = procesa 1 de cada 2 (duplica FPS si el video duplica frames)
WRITE_VIDEO = False     # False evita codificar video aquí (postprocess_ball.py ya genera video_out_clean.mp4)
LOG_INTERVAL = 20       # Mostrar progreso en consola cada N frames procesados


class ThreadedOpenCVReader:
    """Lee frames de video en un hilo secundario usando una cola acotada."""

    def __init__(self, video_path: str, max_frames: int | None = None, stride: int = 1, queue_size: int = 64) -> None:
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")
        self.max_frames = max_frames
        self.stride = max(1, stride)
        self.queue: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        frame_idx = 0
        while not self._stop_event.is_set():
            if self.max_frames is not None and frame_idx >= self.max_frames:
                break
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % self.stride == 0:
                while not self._stop_event.is_set():
                    try:
                        self.queue.put((frame_idx, frame), timeout=0.1)
                        break
                    except queue.Full:
                        continue
            frame_idx += 1
        try:
            self.queue.put(None, timeout=0.1)
        except queue.Full:
            pass

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

    def stop(self) -> None:
        self._stop_event.set()
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


model = YOLO(MODEL_PATH, task="detect")

box_annotator = sv.BoxAnnotator() if WRITE_VIDEO else None
video_info = sv.VideoInfo.from_video_path(VIDEO_INPUT)

total_frames = video_info.total_frames
if MAX_FRAMES is not None:
    total_frames = min(total_frames, MAX_FRAMES) if total_frames else MAX_FRAMES

processed_frames = 0
frames_with_ball = 0
start_time = time.perf_counter()

sink = sv.VideoSink(VIDEO_OUTPUT, video_info=video_info) if WRITE_VIDEO else None
if sink is not None:
    sink.__enter__()

try:
    with ThreadedOpenCVReader(VIDEO_INPUT, max_frames=MAX_FRAMES, stride=FRAME_STRIDE) as reader, open(
        CSV_OUTPUT, mode="w", newline="", encoding="utf-8"
    ) as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "class_id", "confidence", "x1", "y1", "x2", "y2", "cx", "cy", "width", "height"])

        for frame_idx, frame in reader:
            result = model(
                frame,
                imgsz=IMGSZ_NORMAL,
                conf=CONF_NORMAL,
                classes=[BALL_CLASS_ID],
                device=DEVICE,
                verbose=False,
            )[0]
            detections = sv.Detections.from_ultralytics(result)

            if len(detections) > 0:
                frames_with_ball += 1

            for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
                cls_id = int(detections.class_id[i]) if detections.class_id is not None else -1
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                csv_writer.writerow([
                    frame_idx,
                    cls_id,
                    f"{conf:.4f}",
                    f"{x1:.2f}",
                    f"{y1:.2f}",
                    f"{x2:.2f}",
                    f"{y2:.2f}",
                    f"{cx:.2f}",
                    f"{cy:.2f}",
                    f"{w:.2f}",
                    f"{h:.2f}",
                ])

            if sink is not None and box_annotator is not None:
                annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                sink.write_frame(annotated_frame)

            processed_frames += 1
            if processed_frames % LOG_INTERVAL == 0 or (total_frames and frame_idx + 1 >= total_frames):
                elapsed = time.perf_counter() - start_time
                fps = processed_frames / elapsed if elapsed > 0 else 0.0
                effective_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
                if total_frames:
                    pct = ((frame_idx + 1) / total_frames) * 100
                    eta_sec = (total_frames - (frame_idx + 1)) / effective_fps if effective_fps > 0 else 0.0
                    print(
                        f"Frame {frame_idx + 1}/{total_frames} ({pct:.1f}%) "
                        f"| {fps:.2f} FPS (eff: {effective_fps:.2f}) | ETA: {eta_sec:.0f}s | Balón: {frames_with_ball}",
                        flush=True,
                    )
                else:
                    print(
                        f"Frame {frame_idx + 1} | {fps:.2f} FPS | Balón: {frames_with_ball}",
                        flush=True,
                    )
finally:
    if sink is not None:
        sink.__exit__(None, None, None)

print(
    f"Proceso baseline terminado. Frames procesados: {processed_frames}. "
    f"Frames con balón: {frames_with_ball}."
)
