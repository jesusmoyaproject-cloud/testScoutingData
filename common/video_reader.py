import queue
import threading
import cv2
import numpy as np


class ThreadedOpenCVReader:
    """Lee frames de video en un hilo secundario usando una cola acotada."""

    def __init__(
        self,
        video_path: str,
        max_frames: int | None = None,
        stride: int = 1,
        queue_size: int = 64,
    ) -> None:
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
