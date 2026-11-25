import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import supervision as sv
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linear_sum_assignment


@dataclass
class PlayerDetection:
    frame: int
    class_id: int
    confidence: float
    cx: float
    cy: float
    width: float
    height: float
    track_id: int = -1
    source: str = "observed"

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        hw = self.width / 2.0
        hh = self.height / 2.0
        return (self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh)


class KalmanPlayer2D:
    """Filtro de Kalman 2D con estado [cx, cy, vx, vy] para gating y persistencia de IDs."""

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
        Q = np.diag([4.0 * dt, 4.0 * dt, 10.0 * dt, 10.0 * dt])
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
        self.P = (np.eye(4, dtype=np.float64) - K @ self.H) @ self.P


def _bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def _load_and_dedup(
    csv_path: Path,
    classes: list[int] | None,
    exclude_classes: list[int] | None,
    max_frames: int | None,
    nms_dist_px: float,
    nms_iou_thresh: float,
) -> tuple[dict[int, list[PlayerDetection]], int]:
    """Carga el CSV crudo de jugadores y elimina cajas duplicadas por distancia/IoU."""
    allowed_classes = set(int(c) for c in classes) if classes is not None else None
    excluded = set(int(c) for c in exclude_classes) if exclude_classes is not None else set()
    by_frame: dict[int, list[PlayerDetection]] = {}

    with open(csv_path, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls_id = int(row["class_id"])
            if cls_id in excluded:
                continue
            if allowed_classes is not None and cls_id not in allowed_classes:
                continue
            frame = int(row["frame"])
            if max_frames is not None and frame >= max_frames:
                continue
            by_frame.setdefault(frame, []).append(
                PlayerDetection(
                    frame=frame,
                    class_id=cls_id,
                    confidence=float(row["confidence"]),
                    cx=float(row["cx"]),
                    cy=float(row["cy"]),
                    width=float(row["width"]),
                    height=float(row["height"]),
                )
            )

    deduped: dict[int, list[PlayerDetection]] = {}
    removed_dups = 0
    for frame, dets in by_frame.items():
        kept: list[PlayerDetection] = []
        for d in sorted(dets, key=lambda x: x.confidence, reverse=True):
            is_dup = any(
                np.hypot(d.cx - k.cx, d.cy - k.cy) < nms_dist_px or _bbox_iou(d.xyxy, k.xyxy) >= nms_iou_thresh
                for k in kept
            )
            if is_dup:
                removed_dups += 1
                continue
            kept.append(d)
        deduped[frame] = kept

    return deduped, removed_dups


def _match_tracks_to_dets(
    active_tracks: list[tuple[int, list[PlayerDetection], KalmanPlayer2D]],
    track_indices: list[int],
    dets: list[PlayerDetection],
    frame: int,
    base_gate: float,
    max_speed: float,
    maha_thresh: float = 25.0,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Asocia óptimamente tracks activos con detecciones usando Kalman + Hungarian."""
    if not track_indices or not dets:
        return [], set(), set()

    n_t = len(track_indices)
    n_d = len(dets)
    invalid_cost = 1e9
    cost = np.full((n_t, n_d), invalid_cost, dtype=np.float64)

    for r, t_idx in enumerate(track_indices):
        _, t_dets, kf = active_tracks[t_idx]
        last_det = t_dets[-1]
        dt = frame - last_det.frame
        if dt <= 0:
            continue

        probe = KalmanPlayer2D(kf.x[0], kf.x[1])
        probe.x = kf.x.copy()
        probe.P = kf.P.copy()
        pred_x, _ = probe.predict(dt=dt)

        max_euclid = base_gate + max_speed * dt
        pred_box = (
            pred_x[0] - last_det.width / 2.0,
            pred_x[1] - last_det.height / 2.0,
            pred_x[0] + last_det.width / 2.0,
            pred_x[1] + last_det.height / 2.0,
        )

        for c, d in enumerate(dets):
            euclid = float(np.hypot(d.cx - pred_x[0], d.cy - pred_x[1]))
            if euclid > max_euclid:
                continue
            if probe.mahalanobis_sq(d.cx, d.cy) > maha_thresh:
                continue
            iou = _bbox_iou(pred_box, d.xyxy)
            # Combinar distancia normalizada y penalización por falta de IoU
            cost[r, c] = euclid - 25.0 * iou

    row_ind, col_ind = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= invalid_cost * 0.5:
            continue
        t_idx = track_indices[r]
        matches.append((t_idx, c))
        matched_tracks.add(t_idx)
        matched_dets.add(c)

    return matches, matched_tracks, matched_dets


def _build_player_tracklets(
    by_frame: dict[int, list[PlayerDetection]],
    pp: dict[str, Any],
) -> list[list[PlayerDetection]]:
    """Construye tracklets multi-jugador estilo ByteTrack (alta + rescate baja confianza)."""
    high_conf = float(pp.get("high_conf", 0.45))
    low_conf = float(pp.get("low_conf", 0.20))
    max_link_gap = int(pp.get("max_link_gap", 6))
    base_gate = float(pp.get("base_gate_px", 25.0))
    max_speed = float(pp.get("max_speed_px", 8.0))

    active: list[tuple[int, list[PlayerDetection], KalmanPlayer2D]] = []
    finished: list[list[PlayerDetection]] = []
    next_id = 1

    for frame in sorted(by_frame.keys()):
        still_active = []
        for tid, dets, kf in active:
            if frame - dets[-1].frame > max_link_gap:
                finished.append(dets)
            else:
                still_active.append((tid, dets, kf))
        active = still_active

        frame_dets = by_frame[frame]
        high_dets = [d for d in frame_dets if d.confidence >= high_conf]
        low_dets = [d for d in frame_dets if low_conf <= d.confidence < high_conf]

        # Etapa 1: asociar detecciones de alta confianza
        all_t_indices = list(range(len(active)))
        matches_hi, matched_t_hi, matched_d_hi = _match_tracks_to_dets(
            active, all_t_indices, high_dets, frame, base_gate, max_speed
        )
        for t_idx, d_idx in matches_hi:
            tid, dets, kf = active[t_idx]
            dt = frame - dets[-1].frame
            chosen = high_dets[d_idx]
            chosen.track_id = tid
            chosen.source = "observed"
            kf.predict(dt=dt)
            kf.update(chosen.cx, chosen.cy)
            dets.append(chosen)

        # Etapa 2 (ByteTrack): rescatar detecciones de baja confianza para tracks no emparejados
        unmatched_t = [i for i in all_t_indices if i not in matched_t_hi]
        matches_lo, _, _ = _match_tracks_to_dets(
            active, unmatched_t, low_dets, frame, base_gate, max_speed * 0.75
        )
        for t_idx, d_idx in matches_lo:
            tid, dets, kf = active[t_idx]
            dt = frame - dets[-1].frame
            chosen = low_dets[d_idx]
            chosen.track_id = tid
            chosen.source = "rescued_low_conf"
            kf.predict(dt=dt)
            kf.update(chosen.cx, chosen.cy)
            dets.append(chosen)

        # Iniciar nuevos tracks con detecciones de alta confianza no asociadas
        for d_idx, d in enumerate(high_dets):
            if d_idx not in matched_d_hi:
                d.track_id = next_id
                d.source = "observed"
                active.append((next_id, [d], KalmanPlayer2D(d.cx, d.cy)))
                next_id += 1

    finished.extend(dets for _, dets, _ in active)
    return finished


def _is_static(track: list[PlayerDetection], min_frames: int, max_disp: float) -> bool:
    if len(track) < min_frames:
        return False
    xs = [d.cx for d in track]
    ys = [d.cy for d in track]
    return float(np.hypot(max(xs) - min(xs), max(ys) - min(ys))) < max_disp


def _estimate_end_velocity(track: list[PlayerDetection], window: int = 5) -> tuple[float, float]:
    if len(track) < 2:
        return (0.0, 0.0)
    tail = track[-min(len(track), window) :]
    dt = tail[-1].frame - tail[0].frame
    if dt <= 0:
        return (0.0, 0.0)
    return ((tail[-1].cx - tail[0].cx) / dt, (tail[-1].cy - tail[0].cy) / dt)


def _filter_and_stitch_tracklets(
    tracklets: list[list[PlayerDetection]],
    pp: dict[str, Any],
) -> tuple[list[list[PlayerDetection]], int, int]:
    """Filtra ruido/falsos positivos estáticos y re-enlaza tracklets fragmentados."""
    min_track_frames = int(pp.get("min_track_frames", 3))
    s_min = int(pp.get("static_min_frames", 45))
    s_disp = float(pp.get("static_max_disp_px", 8.0))
    max_stitch_gap = int(pp.get("max_stitch_gap", min(35, int(pp.get("max_interp_gap", 35)))))
    base_gate = float(pp.get("base_gate_px", 25.0))
    max_speed = float(pp.get("max_speed_px", 8.0))
    max_stitch_dist = float(pp.get("max_stitch_dist_px", 120.0))

    valid: list[list[PlayerDetection]] = []
    removed_short = 0
    removed_static = 0

    for t in tracklets:
        if len(t) < min_track_frames:
            removed_short += 1
            continue
        if _is_static(t, s_min, s_disp):
            removed_static += 1
            continue
        valid.append(t)

    if not valid:
        return [], removed_short, removed_static

    valid.sort(key=lambda t: (t[0].frame, -len(t)))

    # Re-enlace iterativo entre el final de un tracklet y el inicio de otro cercano
    stitched: list[list[PlayerDetection]] = []
    for curr in valid:
        best_idx = None
        best_dist = float("inf")
        c_start = curr[0]

        for idx, prev in enumerate(stitched):
            p_end = prev[-1]
            gap = c_start.frame - p_end.frame
            if gap <= 0 or gap > max_stitch_gap:
                continue

            # Verificar compatibilidad de tamaño de caja
            w_ratio = c_start.width / max(1.0, p_end.width)
            h_ratio = c_start.height / max(1.0, p_end.height)
            if not (0.5 <= w_ratio <= 2.0 and 0.5 <= h_ratio <= 2.0):
                continue

            vx, vy = _estimate_end_velocity(prev)
            pred_cx = p_end.cx + vx * gap * 0.7
            pred_cy = p_end.cy + vy * gap * 0.7
            dist = float(np.hypot(c_start.cx - pred_cx, c_start.cy - pred_cy))
            raw_dist = float(np.hypot(c_start.cx - p_end.cx, c_start.cy - p_end.cy))
            max_allowed = min(max_stitch_dist, base_gate + max_speed * gap)
            if dist <= max_allowed and raw_dist <= max_stitch_dist and dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is not None:
            stitched[best_idx].extend(curr)
        else:
            stitched.append(list(curr))

    # Reasignar track_id secuencial (1..N) y consistente en todas las detecciones
    stitched.sort(key=lambda t: (t[0].frame, -len(t)))
    for new_id, track in enumerate(stitched, start=1):
        track.sort(key=lambda d: d.frame)
        for d in track:
            d.track_id = new_id

    return stitched, removed_short, removed_static


def _split_impossible_jumps(
    tracks: list[list[PlayerDetection]],
    pp: dict[str, Any],
) -> list[list[PlayerDetection]]:
    """Divide cualquier track si aparece un salto físicamente imposible (evita vuelos por la cancha)."""
    base_gate = float(pp.get("base_gate_px", 25.0))
    max_speed = float(pp.get("max_speed_px", 8.0))
    max_jump_cap = float(pp.get("max_stitch_dist_px", 120.0))
    min_track_frames = int(pp.get("min_track_frames", 3))

    split_tracks: list[list[PlayerDetection]] = []
    for track in tracks:
        if not track:
            continue
        track.sort(key=lambda d: d.frame)
        current = [track[0]]
        for d in track[1:]:
            prev = current[-1]
            dt = max(1, d.frame - prev.frame)
            dist = float(np.hypot(d.cx - prev.cx, d.cy - prev.cy))
            max_allowed = min(max_jump_cap, base_gate + max_speed * dt)
            if dist > max_allowed:
                if len(current) >= min_track_frames:
                    split_tracks.append(current)
                current = [d]
            else:
                current.append(d)
        if len(current) >= min_track_frames:
            split_tracks.append(current)

    split_tracks.sort(key=lambda t: (t[0].frame, -len(t)))
    for new_id, track in enumerate(split_tracks, start=1):
        # Asignar la clase dominante por suma de confianza (evita parpadeo goalkeeper/player)
        cls_scores: dict[int, float] = {}
        for d in track:
            cls_scores[d.class_id] = cls_scores.get(d.class_id, 0.0) + max(0.1, d.confidence)
        dom_cls = max(cls_scores, key=cls_scores.get)
        for d in track:
            d.track_id = new_id
            d.class_id = dom_cls
    return split_tracks


def _suppress_overlapping_tracks(
    tracks: list[list[PlayerDetection]],
    nms_dist_px: float,
    nms_iou_thresh: float,
) -> tuple[list[list[PlayerDetection]], int]:
    """Elimina tracks duplicados que se solapan mayormente sobre otro track más largo."""
    if len(tracks) <= 1:
        return tracks, 0

    # Ordenar de más largo a más corto para conservar siempre el track principal
    ordered = sorted(tracks, key=len, reverse=True)
    kept: list[list[PlayerDetection]] = []
    kept_by_frame: list[dict[int, PlayerDetection]] = []
    removed = 0

    for track in ordered:
        t_map = {d.frame: d for d in track}
        is_dup = False
        for k_map in kept_by_frame:
            common_frames = [f for f in t_map if f in k_map]
            if not common_frames:
                continue
            overlap_hits = sum(
                1
                for f in common_frames
                if np.hypot(t_map[f].cx - k_map[f].cx, t_map[f].cy - k_map[f].cy) < nms_dist_px
                or _bbox_iou(t_map[f].xyxy, k_map[f].xyxy) >= nms_iou_thresh
            )
            if overlap_hits >= max(2, int(0.5 * len(track))):
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            kept.append(track)
            kept_by_frame.append(t_map)

    kept.sort(key=lambda t: (t[0].frame, -len(t)))
    for new_id, track in enumerate(kept, start=1):
        for d in track:
            d.track_id = new_id
    return kept, removed


def _interpolate_tracks_pchip(
    tracks: list[list[PlayerDetection]],
    total_frames: int,
    pp: dict[str, Any],
) -> list[PlayerDetection]:
    """Interpola huecos internos con PCHIP por cada track_id para evitar parpadeos."""
    max_gap = int(pp.get("max_interp_gap", 35))
    max_tail = int(pp.get("max_tail_extrap", 0))

    all_dets: list[PlayerDetection] = []
    for track in tracks:
        if not track:
            continue
        by_f = {d.frame: d for d in track}
        tid = track[0].track_id
        cls_id = track[0].class_id

        obs_f = np.array(sorted(by_f.keys()), dtype=np.float64)
        obs_x = np.array([by_f[int(f)].cx for f in obs_f], dtype=np.float64)
        obs_y = np.array([by_f[int(f)].cy for f in obs_f], dtype=np.float64)
        obs_w = np.array([by_f[int(f)].width for f in obs_f], dtype=np.float64)
        obs_h = np.array([by_f[int(f)].height for f in obs_f], dtype=np.float64)

        if len(obs_f) >= 2:
            px, py = PchipInterpolator(obs_f, obs_x), PchipInterpolator(obs_f, obs_y)
            pw, ph = PchipInterpolator(obs_f, obs_w), PchipInterpolator(obs_f, obs_h)
            for i in range(len(obs_f) - 1):
                f1, f2 = int(obs_f[i]), int(obs_f[i + 1])
                if 1 < f2 - f1 <= max_gap:
                    for f in range(f1 + 1, f2):
                        by_f[f] = PlayerDetection(
                            frame=f,
                            class_id=cls_id,
                            confidence=0.0,
                            cx=float(px(f)),
                            cy=float(py(f)),
                            width=float(max(4.0, pw(f))),
                            height=float(max(4.0, ph(f))),
                            track_id=tid,
                            source="interpolated",
                        )

            last_f = int(obs_f[-1])
            if max_tail > 0 and last_f < total_frames - 1:
                dt = max(1, last_f - int(obs_f[-2]))
                vx, vy = (obs_x[-1] - obs_x[-2]) / dt, (obs_y[-1] - obs_y[-2]) / dt
                cx, cy = float(obs_x[-1]), float(obs_y[-1])
                w, h = float(obs_w[-1]), float(obs_h[-1])
                for step in range(1, min(max_tail, total_frames - 1 - last_f) + 1):
                    vx *= 0.85
                    vy *= 0.85
                    cx += vx
                    cy += vy
                    f = last_f + step
                    by_f[f] = PlayerDetection(
                        frame=f,
                        class_id=cls_id,
                        confidence=0.0,
                        cx=cx,
                        cy=cy,
                        width=w,
                        height=h,
                        track_id=tid,
                        source="extrapolated",
                    )

        all_dets.extend(by_f.values())

    all_dets.sort(key=lambda d: (d.frame, d.track_id))
    return all_dets


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    """Color determinista en BGR por track_id para mantener identidad visual."""
    hue = (track_id * 47) % 180
    hsv = np.uint8([[[hue, 220, 235]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _render_clean_video(
    dets_by_frame: dict[int, list[PlayerDetection]],
    video_path: str,
    video_out: Path,
    max_frames: int | None,
    total_frames: int,
) -> None:
    video_info = sv.VideoInfo.from_video_path(video_path)
    frames_gen = sv.get_video_frames_generator(video_path)
    start_t = time.perf_counter()

    with sv.VideoSink(str(video_out), video_info=video_info) as sink:
        for frame_idx, frame in enumerate(frames_gen):
            if max_frames is not None and frame_idx >= max_frames:
                break
            ann = frame.copy()
            cv2.rectangle(ann, (12, 12), (240, 48), (0, 0, 0), -1)
            cv2.putText(
                ann,
                f"Frame: {frame_idx}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for d in dets_by_frame.get(frame_idx, []):
                x1, y1, x2, y2 = [int(round(v)) for v in d.xyxy]
                color = _color_for_track(d.track_id)
                thickness = 2 if d.source in ("observed", "rescued_low_conf") else 1
                cv2.rectangle(ann, (x1, y1), (x2, y2), color, thickness)
                suffix = f"{d.confidence:.2f}" if d.confidence > 0 else d.source[:6]
                lbl = f"ID:{d.track_id} {suffix}"
                cv2.putText(
                    ann,
                    lbl,
                    (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            sink.write_frame(ann)
            if (frame_idx + 1) % 200 == 0 or (total_frames and frame_idx + 1 == total_frames):
                fps = (frame_idx + 1) / max(1e-6, time.perf_counter() - start_t)
                print(f"[PLAYER-RENDER] {frame_idx + 1}/{total_frames} | {fps:.1f} FPS", flush=True)


def run_player_postprocess(cfg: dict[str, Any], csv_input: Path) -> dict[str, Any]:
    """Asigna IDs persistentes, rescata baja confianza e interpola huecos con PCHIP."""
    out_dir = Path(cfg["task_output_dir"])
    video_path = cfg["video"]
    inf_cfg = cfg["inference"]
    player_cfg = cfg.get("player", {})
    pp = player_cfg.get("postprocess", {})

    classes = player_cfg.get("classes")
    exclude_classes = player_cfg.get("exclude_classes", [0])
    max_frames = inf_cfg.get("max_frames")
    video_info = sv.VideoInfo.from_video_path(video_path)
    total_frames = (
        min(video_info.total_frames, max_frames)
        if (max_frames and video_info.total_frames)
        else (max_frames or video_info.total_frames)
    )

    by_frame, removed_dups = _load_and_dedup(
        csv_input,
        classes=classes,
        exclude_classes=exclude_classes,
        max_frames=max_frames,
        nms_dist_px=float(pp.get("nms_dist_px", 20.0)),
        nms_iou_thresh=float(pp.get("nms_iou_thresh", 0.7)),
    )
    tracklets = _build_player_tracklets(by_frame, pp)
    tracks, removed_short, removed_static = _filter_and_stitch_tracklets(tracklets, pp)
    tracks = _split_impossible_jumps(tracks, pp)
    tracks, removed_overlap_tracks = _suppress_overlapping_tracks(
        tracks,
        nms_dist_px=float(pp.get("nms_dist_px", 20.0)),
        nms_iou_thresh=float(pp.get("nms_iou_thresh", 0.7)),
    )
    final_dets = _interpolate_tracks_pchip(tracks, total_frames, pp)

    csv_clean = out_dir / "detections_player_clean.csv"
    with open(csv_clean, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame", "track_id", "class_id", "source", "confidence", "x1", "y1", "x2", "y2", "cx", "cy", "width", "height"]
        )
        for d in final_dets:
            x1, y1, x2, y2 = d.xyxy
            writer.writerow([
                d.frame,
                d.track_id,
                d.class_id,
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

    counts = {"observed": 0, "rescued_low_conf": 0, "interpolated": 0, "extrapolated": 0}
    dets_by_frame: dict[int, list[PlayerDetection]] = {}
    for d in final_dets:
        counts[d.source] = counts.get(d.source, 0) + 1
        dets_by_frame.setdefault(d.frame, []).append(d)

    if bool(pp.get("render_video", False)):
        profile_name = cfg.get("name", "run")
        video_clean = Path(cfg.get("videos_output_dir", out_dir)) / f"player_{profile_name}_clean.mp4"
        _render_clean_video(dets_by_frame, video_path, video_clean, max_frames, total_frames)

    return {
        "csv_clean": str(csv_clean),
        "unique_tracks": len(tracks),
        "total_detections_clean": len(final_dets),
        "frames_with_players": len(dets_by_frame),
        "removed_dups": removed_dups,
        "removed_short_tracks": removed_short,
        "removed_static_tracks": removed_static,
        "removed_overlap_tracks": removed_overlap_tracks,
        "sources": counts,
    }
