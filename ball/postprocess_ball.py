import csv
import time
from dataclasses import dataclass
import cv2
import numpy as np
import supervision as sv
from scipy.interpolate import PchipInterpolator

# --- Configuración ---
CSV_INPUT = "G:/Mi unidad/utp clases/2026-3/Aquitectura Orientada al Servicio/Semana 15/proyectoSOA/scouringData_VideoInference/ball/detections_baseline.csv"
CSV_OUTPUT = "G:/Mi unidad/utp clases/2026-3/Aquitectura Orientada al Servicio/Semana 15/proyectoSOA/scouringData_VideoInference/ball/detections_clean.csv"
VIDEO_INPUT = "G:/Mi unidad/ScoutingData/videosPartidos/ccVSutc_liga1max_2026_clausura_st.mp4"
VIDEO_OUTPUT = "G:/Mi unidad/utp clases/2026-3/Aquitectura Orientada al Servicio/Semana 15/proyectoSOA/scouringData_VideoInference/ball/video_out_clean.mp4"

MAX_FRAMES = 5000         # Mismo rango que main_baseline.py (None para todo el video)
BALL_CLASS_ID = 0

# Parámetros estilo ByteTrack + Kalman + PCHIP
HIGH_CONF = 0.45          # Confianza mínima para iniciar/anclar tracklets
LOW_CONF = 0.20           # Confianza mínima para rescate dentro de la puerta cinemática
NMS_DIST_PX = 20.0        # Elimina doble caja sobre el mismo balón
MAX_LINK_GAP = 4          # Hueco máximo para unir dentro de un mismo tracklet
BASE_GATE_PX = 25.0       # Radio base de aceptación (px)
MAX_SPEED_PX = 25.0       # Velocidad máxima esperada del balón (px/frame)
STATIC_MIN_FRAMES = 15    # Tracklets con >= N frames y desplazamiento mínimo se marcan estáticos
STATIC_MAX_DISP_PX = 12.0 # Desplazamiento máximo total para considerar un objeto estático
MAX_INTERP_GAP = 75       # Hueco máximo para rescatar/interpolar con PCHIP (~1.25s a 60fps)
MAX_TAIL_EXTRAP = 5       # Máximo de frames extrapolados al final si no hay detección futura
RENDER_VIDEO = True       # True para generar el video comparativo video_out_clean.mp4


@dataclass
class Detection:
    frame: int
    class_id: int
    confidence: float
    cx: float
    cy: float
    width: float
    height: float
    source: str = "observed"

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        hw = self.width / 2.0
        hh = self.height / 2.0
        return (self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh)


class KalmanBall2D:
    """Filtro de Kalman 2D con estado [cx, cy, vx, vy] para gating y suavizado."""

    def __init__(self, cx: float, cy: float) -> None:
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([25.0, 25.0, 100.0, 100.0])
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self.R = np.diag([16.0, 16.0])

    def predict(self, dt: int = 1) -> tuple[np.ndarray, np.ndarray]:
        F = np.array(
            [
                [1.0, 0.0, float(dt), 0.0],
                [0.0, 1.0, 0.0, float(dt)],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q_pos = 4.0 * dt
        q_vel = 12.0 * dt
        Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x.copy(), self.P.copy()

    def mahalanobis_sq(self, cx: float, cy: float) -> float:
        z = np.array([cx, cy], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)

    def update(self, cx: float, cy: float) -> None:
        z = np.array([cx, cy], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P


def load_and_deduplicate(csv_path: str) -> tuple[dict[int, list[Detection]], int]:
    """Carga el CSV y aplica NMS por distancia para eliminar cajas duplicadas en un mismo balón."""
    by_frame: dict[int, list[Detection]] = {}
    raw_count = 0

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_id = int(row["class_id"])
            if cls_id != BALL_CLASS_ID:
                continue
            frame = int(row["frame"])
            if MAX_FRAMES is not None and frame >= MAX_FRAMES:
                continue
            raw_count += 1
            det = Detection(
                frame=frame,
                class_id=cls_id,
                confidence=float(row["confidence"]),
                cx=float(row["cx"]),
                cy=float(row["cy"]),
                width=float(row["width"]),
                height=float(row["height"]),
            )
            by_frame.setdefault(frame, []).append(det)

    # NMS por distancia entre centros: conserva la de mayor confianza
    deduped: dict[int, list[Detection]] = {}
    removed_dups = 0
    for frame, dets in by_frame.items():
        dets_sorted = sorted(dets, key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []
        for d in dets_sorted:
            if any(np.hypot(d.cx - k.cx, d.cy - k.cy) < NMS_DIST_PX for k in kept):
                removed_dups += 1
                continue
            kept.append(d)
        deduped[frame] = kept

    return deduped, removed_dups


def build_high_conf_tracklets(by_frame: dict[int, list[Detection]]) -> list[list[Detection]]:
    """Agrupa detecciones de alta confianza en tracklets continuos usando Kalman + gating."""
    active_tracks: list[tuple[list[Detection], KalmanBall2D]] = []
    finished_tracks: list[list[Detection]] = []

    all_frames = sorted(by_frame.keys())
    if not all_frames:
        return []

    for frame in all_frames:
        # Cerrar tracks cuyo hueco excede MAX_LINK_GAP
        still_active: list[tuple[list[Detection], KalmanBall2D]] = []
        for dets, kf in active_tracks:
            if frame - dets[-1].frame > MAX_LINK_GAP:
                finished_tracks.append(dets)
            else:
                still_active.append((dets, kf))
        active_tracks = still_active

        high_dets = [d for d in by_frame[frame] if d.confidence >= HIGH_CONF]
        used_dets = set()

        # Asociar detecciones a los tracks activos más cercanos en gating
        for track_idx, (dets, kf) in enumerate(active_tracks):
            dt = frame - dets[-1].frame
            # Copiamos predicción sin mutar hasta confirmar match, o predecimos paso a paso
            kf_probe = KalmanBall2D(kf.x[0], kf.x[1])
            kf_probe.x = kf.x.copy()
            kf_probe.P = kf.P.copy()
            pred_x, _ = kf_probe.predict(dt=dt)

            best_i = None
            best_score = float("inf")
            max_euclid = BASE_GATE_PX + MAX_SPEED_PX * dt

            for i, d in enumerate(high_dets):
                if i in used_dets:
                    continue
                euclid = float(np.hypot(d.cx - pred_x[0], d.cy - pred_x[1]))
                if euclid > max_euclid:
                    continue
                maha = kf_probe.mahalanobis_sq(d.cx, d.cy)
                if maha <= 25.0 and euclid < best_score:
                    best_score = euclid
                    best_i = i

            if best_i is not None:
                chosen = high_dets[best_i]
                used_dets.add(best_i)
                kf.predict(dt=dt)
                kf.update(chosen.cx, chosen.cy)
                dets.append(chosen)

        # Iniciar nuevos tracks con detecciones no asociadas
        for i, d in enumerate(high_dets):
            if i not in used_dets:
                active_tracks.append(([d], KalmanBall2D(d.cx, d.cy)))

    for dets, _ in active_tracks:
        finished_tracks.append(dets)

    return finished_tracks


def is_static_tracklet(track: list[Detection]) -> bool:
    """Detecta falsos positivos estáticos (logos, balones fuera de juego)."""
    if len(track) < STATIC_MIN_FRAMES:
        return False
    xs = [d.cx for d in track]
    ys = [d.cy for d in track]
    disp = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
    return disp < STATIC_MAX_DISP_PX


def select_global_trajectory(tracklets: list[list[Detection]]) -> list[list[Detection]]:
    """
    Programación dinámica (estilo Viterbi) para elegir la secuencia óptima de tracklets
    que maximiza cobertura/confianza y penaliza saltos físicamente imposibles.
    """
    # Filtrar tracklets muy cortos aislados (< 2 frames) y estáticos
    valid = [t for t in tracklets if len(t) >= 2 and not is_static_tracklet(t)]
    if not valid:
        valid = [t for t in tracklets if not is_static_tracklet(t)]
    if not valid:
        return []

    valid.sort(key=lambda t: (t[0].frame, -len(t)))
    n = len(valid)

    def track_score(t: list[Detection]) -> float:
        return sum(d.confidence for d in t) + 0.5 * len(t)

    dp = [track_score(t) for t in valid]
    parent = [-1] * n

    for i in range(n):
        t_curr = valid[i]
        start_f = t_curr[0].frame
        for j in range(i):
            t_prev = valid[j]
            end_f = t_prev[-1].frame
            gap = start_f - end_f
            if gap <= 0:
                continue

            if gap <= MAX_INTERP_GAP:
                dist = float(np.hypot(t_curr[0].cx - t_prev[-1].cx, t_curr[0].cy - t_prev[-1].cy))
                max_allowed = BASE_GATE_PX + MAX_SPEED_PX * gap
                if dist > max_allowed:
                    continue
                transition_penalty = 0.15 * dist
            else:
                # Permitir continuar tras oclusiones largas o cortes de cámara sin descartar el resto del video
                transition_penalty = 15.0

            cand_score = dp[j] + track_score(t_curr) - transition_penalty
            if cand_score > dp[i]:
                dp[i] = cand_score
                parent[i] = j

    best_idx = int(np.argmax(dp))
    chain: list[list[Detection]] = []
    while best_idx != -1:
        chain.append(valid[best_idx])
        best_idx = parent[best_idx]
    chain.reverse()
    return chain


def rescue_low_confidence_in_gaps(
    chain: list[list[Detection]],
    by_frame: dict[int, list[Detection]],
) -> dict[int, Detection]:
    """
    Estilo ByteTrack: rellena huecos con detecciones reales de baja confianza (LOW_CONF..HIGH_CONF)
    únicamente si caen cerca de la línea esperada entre los extremos del hueco.
    """
    selected: dict[int, Detection] = {}
    for t in chain:
        for d in t:
            selected[d.frame] = d

    if not selected:
        return selected

    frames_obs = sorted(selected.keys())
    for idx in range(len(frames_obs) - 1):
        f1 = frames_obs[idx]
        f2 = frames_obs[idx + 1]
        gap = f2 - f1
        if gap <= 1 or gap > MAX_INTERP_GAP:
            continue

        d1 = selected[f1]
        d2 = selected[f2]
        gate_radius = BASE_GATE_PX + 2.0 * min(gap, 15)

        for f in range(f1 + 1, f2):
            if f not in by_frame:
                continue
            alpha = (f - f1) / gap
            exp_x = d1.cx + alpha * (d2.cx - d1.cx)
            exp_y = d1.cy + alpha * (d2.cy - d1.cy)

            best_cand = None
            best_dist = gate_radius
            for cand in by_frame[f]:
                if cand.confidence < LOW_CONF:
                    continue
                dist = float(np.hypot(cand.cx - exp_x, cand.cy - exp_y))
                if dist < best_dist:
                    best_dist = dist
                    best_cand = cand

            if best_cand is not None:
                best_cand.source = "rescued_low_conf"
                selected[f] = best_cand

    return selected


def remove_detour_outliers(
    selected: dict[int, Detection],
    max_detour_len: int = 12,
    jump_thresh_px: float = 50.0,
) -> tuple[dict[int, Detection], int]:
    """Elimina desvíos cortos de ida y vuelta (A -> falso positivo lejano B -> A')."""
    cleaned = dict(selected)
    removed = 0
    changed = True

    while changed:
        changed = False
        frames = sorted(cleaned.keys())
        n = len(frames)
        for i in range(1, n - 1):
            f_prev = frames[i - 1]
            f_start = frames[i]
            d_prev = cleaned[f_prev]
            d_start = cleaned[f_start]

            if np.hypot(d_start.cx - d_prev.cx, d_start.cy - d_prev.cy) <= jump_thresh_px:
                continue

            # Buscar si dentro de max_detour_len observaciones vuelve cerca de la línea base d_prev -> d_after
            for k in range(1, max_detour_len + 1):
                if i + k >= n:
                    break
                f_after = frames[i + k]
                if f_after - f_prev > MAX_INTERP_GAP:
                    break
                d_after = cleaned[f_after]
                d_end = cleaned[frames[i + k - 1]]

                jump_back = np.hypot(d_after.cx - d_end.cx, d_after.cy - d_end.cy)
                base_span = np.hypot(d_after.cx - d_prev.cx, d_after.cy - d_prev.cy)
                # Si vuelve a estar cerca de d_prev (o mucho más cerca que el desvío)
                if jump_back > jump_thresh_px and base_span < jump_thresh_px + 2.5 * (f_after - f_prev):
                    for rm_idx in range(i, i + k):
                        cleaned.pop(frames[rm_idx], None)
                        removed += 1
                    changed = True
                    break
            if changed:
                break

    return cleaned, removed


def interpolate_and_smooth(
    selected: dict[int, Detection],
    total_frames: int,
) -> dict[int, Detection]:
    """Rellena los huecos internos con PCHIP (sin oscilaciones) y extrapolación corta al final."""
    if not selected:
        return {}

    obs_frames = np.array(sorted(selected.keys()), dtype=np.float64)
    obs_cx = np.array([selected[int(f)].cx for f in obs_frames], dtype=np.float64)
    obs_cy = np.array([selected[int(f)].cy for f in obs_frames], dtype=np.float64)
    obs_w = np.array([selected[int(f)].width for f in obs_frames], dtype=np.float64)
    obs_h = np.array([selected[int(f)].height for f in obs_frames], dtype=np.float64)

    result: dict[int, Detection] = dict(selected)

    if len(obs_frames) >= 2:
        pchip_x = PchipInterpolator(obs_frames, obs_cx)
        pchip_y = PchipInterpolator(obs_frames, obs_cy)
        pchip_w = PchipInterpolator(obs_frames, obs_w)
        pchip_h = PchipInterpolator(obs_frames, obs_h)

        for i in range(len(obs_frames) - 1):
            f1 = int(obs_frames[i])
            f2 = int(obs_frames[i + 1])
            gap = f2 - f1
            if 1 < gap <= MAX_INTERP_GAP:
                for f in range(f1 + 1, f2):
                    cx = float(pchip_x(f))
                    cy = float(pchip_y(f))
                    w = float(max(4.0, pchip_w(f)))
                    h = float(max(4.0, pchip_h(f)))
                    result[f] = Detection(
                        frame=f,
                        class_id=BALL_CLASS_ID,
                        confidence=0.0,
                        cx=cx,
                        cy=cy,
                        width=w,
                        height=h,
                        source="interpolated",
                    )

    # Extrapolación corta amortiguada al final (para no inventar trayectorias largas sin futuro)
    last_f = int(obs_frames[-1])
    if len(obs_frames) >= 2 and last_f < total_frames - 1:
        f_prev = int(obs_frames[-2])
        dt = max(1, last_f - f_prev)
        vx = (obs_cx[-1] - obs_cx[-2]) / dt
        vy = (obs_cy[-1] - obs_cy[-2]) / dt
        cur_x, cur_y = float(obs_cx[-1]), float(obs_cy[-1])
        w, h = float(obs_w[-1]), float(obs_h[-1])
        for step in range(1, min(MAX_TAIL_EXTRAP, total_frames - 1 - last_f) + 1):
            vx *= 0.85
            vy *= 0.85
            cur_x += vx
            cur_y += vy
            f = last_f + step
            result[f] = Detection(
                frame=f,
                class_id=BALL_CLASS_ID,
                confidence=0.0,
                cx=cur_x,
                cy=cur_y,
                width=w,
                height=h,
                source="extrapolated",
            )

    return result


def save_clean_csv(detections: dict[int, Detection], csv_out: str) -> None:
    with open(csv_out, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "source", "confidence", "x1", "y1", "x2", "y2", "cx", "cy", "width", "height"])
        for frame in sorted(detections.keys()):
            d = detections[frame]
            x1, y1, x2, y2 = d.xyxy
            writer.writerow([
                d.frame,
                d.source,
                f"{d.confidence:.4f}",
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{x2:.2f}",
                f"{y2:.2f}",
                f"{d.cx:.2f}",
                f"{d.cy:.2f}",
                f"{d.width:.2f}",
                f"{d.height:.2f}",
            ])


def render_clean_video(detections: dict[int, Detection], video_in: str, video_out: str, max_frames: int | None) -> None:
    video_info = sv.VideoInfo.from_video_path(video_in)
    frames_generator = sv.get_video_frames_generator(video_in)
    total = min(video_info.total_frames, max_frames) if (max_frames and video_info.total_frames) else (max_frames or video_info.total_frames)

    start_time = time.perf_counter()
    with sv.VideoSink(video_out, video_info=video_info) as sink:
        for frame_idx, frame in enumerate(frames_generator):
            if max_frames is not None and frame_idx >= max_frames:
                break

            annotated = frame.copy()
            cv2.rectangle(annotated, (12, 12), (210, 48), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                f"Frame: {frame_idx}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            d = detections.get(frame_idx)
            if d is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in d.xyxy]
                # Verde = observado/rescatado, Amarillo = interpolado/extrapolado
                color = (0, 255, 0) if d.source in ("observed", "rescued_low_conf") else (0, 220, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{d.source}" if d.confidence == 0 else f"{d.source} {d.confidence:.2f}"
                cv2.putText(annotated, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            sink.write_frame(annotated)

            if (frame_idx + 1) % 100 == 0 or (total and frame_idx + 1 == total):
                elapsed = time.perf_counter() - start_time
                fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
                print(f"Renderizando video limpio: {frame_idx + 1}/{total} frames | {fps:.1f} FPS", flush=True)


def main() -> None:
    video_info = sv.VideoInfo.from_video_path(VIDEO_INPUT)
    total_frames = min(video_info.total_frames, MAX_FRAMES) if (MAX_FRAMES and video_info.total_frames) else (MAX_FRAMES or video_info.total_frames)

    by_frame, removed_dups = load_and_deduplicate(CSV_INPUT)
    tracklets = build_high_conf_tracklets(by_frame)
    chain = select_global_trajectory(tracklets)
    selected = rescue_low_confidence_in_gaps(chain, by_frame)
    selected, detours_removed = remove_detour_outliers(selected)
    final_detections = interpolate_and_smooth(selected, total_frames=total_frames)

    save_clean_csv(final_detections, CSV_OUTPUT)

    counts = {"observed": 0, "rescued_low_conf": 0, "interpolated": 0, "extrapolated": 0}
    for d in final_detections.values():
        counts[d.source] = counts.get(d.source, 0) + 1

    print(f"--- Resumen Post-Procesamiento ({total_frames} frames) ---")
    print(f"Cajas duplicadas eliminadas (NMS): {removed_dups}")
    print(f"Desvíos bruscos (ida y vuelta) eliminados: {detours_removed}")
    print(f"Tracklets encontrados: {len(tracklets)} | Seleccionados en cadena global: {len(chain)}")
    print(f"Frames con exactamente 1 balón: {len(final_detections)}/{total_frames}")
    print(f"Desglose: {counts}")
    print(f"CSV limpio guardado en: {CSV_OUTPUT}")

    if RENDER_VIDEO:
        render_clean_video(final_detections, VIDEO_INPUT, VIDEO_OUTPUT, MAX_FRAMES)
        print(f"Video limpio guardado en: {VIDEO_OUTPUT}")


if __name__ == "__main__":
    main()
