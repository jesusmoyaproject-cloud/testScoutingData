"""Estabilización temporal y geométrica de keypoints y homografía (incluye EMA doble pasado)."""

from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np
from homography.geometry import (
    compute_distances_m,
    compute_homography,
    project_points,
    reproject_real_to_px,
)


@dataclass
class _KptTrackState:
    x_px: float
    y_px: float
    vx_px: float
    vy_px: float
    last_frame: int
    consecutive_hits: int
    last_conf: float


def _bidirectional_ema_1d(values: np.ndarray, alpha: float) -> np.ndarray:
    """EMA de doble pasado (forward + backward) para suavizado sin desfase (zero-phase)."""
    n = len(values)
    if n <= 1:
        return values.copy()

    fwd = np.empty_like(values, dtype=np.float64)
    fwd[0] = values[0]
    for i in range(1, n):
        fwd[i] = alpha * values[i] + (1.0 - alpha) * fwd[i - 1]

    bwd = np.empty_like(values, dtype=np.float64)
    bwd[-1] = fwd[-1]
    for i in range(n - 2, -1, -1):
        bwd[i] = alpha * fwd[i] + (1.0 - alpha) * bwd[i + 1]

    return bwd


def _bidirectional_ema_matrices(matrices: list[np.ndarray | None], alpha: float) -> list[np.ndarray | None]:
    """Aplica EMA de doble pasado sobre matrices 3x3 normalizadas en tramos continuos."""
    n = len(matrices)
    if n == 0:
        return []

    out: list[np.ndarray | None] = [None] * n
    i = 0
    while i < n:
        if matrices[i] is None:
            i += 1
            continue
        j = i
        while j < n and matrices[j] is not None:
            j += 1

        seg = [m / (m[2, 2] if abs(m[2, 2]) > 1e-8 else 1.0) for m in matrices[i:j]]
        m_len = len(seg)
        if m_len == 1:
            out[i] = seg[0].astype(np.float32)
        else:
            fwd: list[np.ndarray] = [seg[0]]
            for k in range(1, m_len):
                blended = alpha * seg[k] + (1.0 - alpha) * fwd[-1]
                blended /= blended[2, 2] if abs(blended[2, 2]) > 1e-8 else 1.0
                fwd.append(blended)

            bwd: list[np.ndarray] = [fwd[-1]] * m_len
            for k in range(m_len - 2, -1, -1):
                blended = alpha * fwd[k] + (1.0 - alpha) * bwd[k + 1]
                blended /= blended[2, 2] if abs(blended[2, 2]) > 1e-8 else 1.0
                bwd[k] = blended

            for k in range(m_len):
                out[i + k] = bwd[k].astype(np.float32)
        i = j
    return out


class HomographyStabilizer:
    """Filtra saltos/derivas por keypoint en pasada 1 y aplica EMA doble pasado en pasada 2."""

    def __init__(
        self,
        max_jump_px: float = 35.0,
        max_geom_err_m: float = 2.5,
        ema_alpha_min: float = 0.25,
        ema_alpha_max: float = 0.70,
        speed_ref_px: float = 15.0,
        max_hold_frames: int = 2,
        h_ema_alpha: float = 0.45,
        kpt_ema_alpha: float = 0.45,
        max_h_center_jump_m: float = 12.0,
        scene_cut_diff_thresh: float = 35.0,
    ) -> None:
        self.max_jump_px = max_jump_px
        self.max_geom_err_m = max_geom_err_m
        self.ema_alpha_min = ema_alpha_min
        self.ema_alpha_max = ema_alpha_max
        self.speed_ref_px = speed_ref_px
        self.max_hold_frames = max_hold_frames
        self.h_ema_alpha = h_ema_alpha
        self.kpt_ema_alpha = kpt_ema_alpha
        self.max_h_center_jump_m = max_h_center_jump_m
        self.scene_cut_diff_thresh = scene_cut_diff_thresh

        self._tracks: dict[int, _KptTrackState] = {}
        self._prev_gray_small: np.ndarray | None = None
        self._smooth_h: np.ndarray | None = None
        self._last_h_frame: int | None = None

    def reset(self) -> None:
        """Reinicia el estado temporal tras un corte de cámara."""
        self._tracks.clear()
        self._smooth_h = None
        self._last_h_frame = None

    def detect_scene_cut(self, frame: np.ndarray) -> bool:
        """Detecta cortes bruscos de cámara comparando luminancia reducida entre frames."""
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        is_cut = False
        if self._prev_gray_small is not None:
            mad = float(np.mean(cv2.absdiff(gray, self._prev_gray_small)))
            if mad > self.scene_cut_diff_thresh:
                is_cut = True
                self.reset()
        self._prev_gray_small = gray
        return is_cut

    def _geom_error_against_prev_h(self, raw_x: float, raw_y: float, real_x_m: float, real_y_m: float) -> float | None:
        """Mide el error en metros contra la homografía suavizada previa."""
        if self._smooth_h is None:
            return None
        pt = np.array([[[raw_x, raw_y]]], dtype=np.float32)
        proj = cv2.perspectiveTransform(pt, self._smooth_h).reshape(2)
        if not np.all(np.isfinite(proj)):
            return None
        return float(np.hypot(proj[0] - real_x_m, proj[1] - real_y_m))

    def update_keypoints(
        self,
        frame_idx: int,
        raw_records: list[dict[str, Any]],
        specs_by_idx: dict[int, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Pasada 1: filtra derivas geométricas contra H y saltos bruscos en píxeles."""
        active_records: list[dict[str, Any]] = []
        outlier_records: list[dict[str, Any]] = []

        h_recent = self._smooth_h is not None and self._last_h_frame is not None and (frame_idx - self._last_h_frame) <= 5

        for rec in raw_records:
            kpt_id = int(rec["keypoint_idx"])
            raw_x = float(rec["det_x_px"])
            raw_y = float(rec["det_y_px"])
            conf = float(rec["confidence"])

            rec["raw_x_px"] = raw_x
            rec["raw_y_px"] = raw_y
            rec["is_outlier"] = False
            rec["source"] = "detected"

            if h_recent:
                geom_err = self._geom_error_against_prev_h(raw_x, raw_y, rec["real_x_m"], rec["real_y_m"])
                if geom_err is not None and geom_err > self.max_geom_err_m:
                    rec["is_outlier"] = True
                    rec["source"] = "rejected_geom"
                    outlier_records.append(rec)
                    continue

            prev = self._tracks.get(kpt_id)
            if prev is not None:
                dt = max(1, frame_idx - prev.last_frame)
                if dt <= 8:
                    pred_x = prev.x_px + prev.vx_px * min(dt, 2)
                    pred_y = prev.y_px + prev.vy_px * min(dt, 2)
                    jump = float(np.hypot(raw_x - pred_x, raw_y - pred_y))
                    allowed_jump = self.max_jump_px * (1.0 + 0.35 * (dt - 1))
                    if jump > allowed_jump:
                        rec["is_outlier"] = True
                        rec["source"] = "rejected_jump"
                        outlier_records.append(rec)
                        continue

                    speed = float(np.hypot(raw_x - prev.x_px, raw_y - prev.y_px)) / dt
                    ratio = min(1.0, speed / max(1e-3, self.speed_ref_px))
                    alpha = self.ema_alpha_min + ratio * (self.ema_alpha_max - self.ema_alpha_min)
                    smooth_x = alpha * raw_x + (1.0 - alpha) * prev.x_px
                    smooth_y = alpha * raw_y + (1.0 - alpha) * prev.y_px
                    vx = (smooth_x - prev.x_px) / dt
                    vy = (smooth_y - prev.y_px) / dt
                    hits = prev.consecutive_hits + 1
                else:
                    smooth_x, smooth_y, vx, vy, hits = raw_x, raw_y, 0.0, 0.0, 1
            else:
                smooth_x, smooth_y, vx, vy, hits = raw_x, raw_y, 0.0, 0.0, 1

            self._tracks[kpt_id] = _KptTrackState(
                x_px=smooth_x,
                y_px=smooth_y,
                vx_px=vx,
                vy_px=vy,
                last_frame=frame_idx,
                consecutive_hits=hits,
                last_conf=conf,
            )
            rec["det_x_px"] = smooth_x
            rec["det_y_px"] = smooth_y
            active_records.append(rec)

        return active_records, outlier_records

    def stabilize_homography(
        self,
        frame_idx: int,
        candidate_h: np.ndarray | None,
        frame_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """Actualiza referencia causal de H para el gate geométrico de la pasada 1."""
        h_img, w_img = frame_shape
        center_px = np.array([[[w_img / 2.0, h_img / 2.0]]], dtype=np.float32)

        if candidate_h is None:
            if self._smooth_h is not None and self._last_h_frame is not None:
                if frame_idx - self._last_h_frame <= self.max_hold_frames:
                    return self._smooth_h.copy()
            return None

        cand_norm = candidate_h / (candidate_h[2, 2] if abs(candidate_h[2, 2]) > 1e-8 else 1.0)
        if self._smooth_h is not None and self._last_h_frame is not None and (frame_idx - self._last_h_frame) <= 5:
            prev_c = cv2.perspectiveTransform(center_px, self._smooth_h).reshape(2)
            curr_c = cv2.perspectiveTransform(center_px, cand_norm).reshape(2)
            center_shift = float(np.linalg.norm(curr_c - prev_c))
            if center_shift > self.max_h_center_jump_m * max(1, frame_idx - self._last_h_frame):
                return self._smooth_h.copy()

            blended = self.h_ema_alpha * cand_norm + (1.0 - self.h_ema_alpha) * self._smooth_h
            blended /= blended[2, 2] if abs(blended[2, 2]) > 1e-8 else 1.0
            self._smooth_h = blended
        else:
            self._smooth_h = cand_norm

        self._last_h_frame = frame_idx
        return self._smooth_h.copy()

    def apply_double_pass_ema(
        self,
        frames_data: list[dict[str, Any]],
        specs_by_idx: dict[int, Any],
        ransac_thresh_m: float = 2.0,
    ) -> None:
        """Pasada 2: interpola micro-gaps y aplica EMA bidireccional (forward+backward) sin desfase."""
        n = len(frames_data)
        if n == 0:
            return

        # Dividir en segmentos continuos entre cortes de cámara
        segments: list[tuple[int, int]] = []
        seg_start = 0
        for i, fd in enumerate(frames_data):
            if i > 0 and fd.get("scene_cut", False):
                segments.append((seg_start, i))
                seg_start = i
        segments.append((seg_start, n))

        for s_idx, e_idx in segments:
            self._smooth_segment_keypoints(frames_data, s_idx, e_idx, specs_by_idx)

            # Recalcular H candidato con keypoints suavizados por doble pasado
            cand_hs: list[np.ndarray | None] = []
            for i in range(s_idx, e_idx):
                active = frames_data[i]["active_records"]
                if len(active) >= 4:
                    src_px = np.array([[r["det_x_px"], r["det_y_px"]] for r in active], dtype=np.float32)
                    dst_m = np.array([[r["real_x_m"], r["real_y_m"]] for r in active], dtype=np.float32)
                    hc, _ = compute_homography(src_px, dst_m, ransac_thresh_m=ransac_thresh_m, refine_inliers=True)
                    cand_hs.append(hc)
                else:
                    cand_hs.append(None)

            # EMA doble pasado sobre la matriz H del segmento
            smooth_hs = _bidirectional_ema_matrices(cand_hs, alpha=self.h_ema_alpha)
            for offset, h_mat in enumerate(smooth_hs):
                fd = frames_data[s_idx + offset]
                fd["h_matrix"] = h_mat
                active = fd["active_records"]
                if h_mat is not None and len(active) > 0:
                    src_px = np.array([[r["det_x_px"], r["det_y_px"]] for r in active], dtype=np.float32)
                    dst_m = np.array([[r["real_x_m"], r["real_y_m"]] for r in active], dtype=np.float32)
                    proj_m = project_points(h_mat, src_px)
                    dists_m = compute_distances_m(proj_m, dst_m)
                    reproj_px = reproject_real_to_px(h_mat, dst_m)
                    for i_r, (r, (px_m, py_m), d_m) in enumerate(zip(active, proj_m, dists_m)):
                        r["proj_x_m"] = float(px_m)
                        r["proj_y_m"] = float(py_m)
                        r["dist_m"] = float(d_m)
                        r["is_inlier"] = bool(d_m <= ransac_thresh_m)
                        r["is_outlier"] = not r["is_inlier"]
                        if reproj_px is not None:
                            r["real_x_px"] = float(reproj_px[i_r, 0])
                            r["real_y_px"] = float(reproj_px[i_r, 1])

    def _smooth_segment_keypoints(
        self,
        frames_data: list[dict[str, Any]],
        s_idx: int,
        e_idx: int,
        specs_by_idx: dict[int, Any],
    ) -> None:
        """Interpola micro-gaps bidireccionalmente y aplica EMA doble pasado por keypoint."""
        by_kpt: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for i in range(s_idx, e_idx):
            for rec in frames_data[i]["active_records"]:
                by_kpt.setdefault(int(rec["keypoint_idx"]), []).append((i, rec))

        for kpt_id, obs_list in by_kpt.items():
            spec = specs_by_idx.get(kpt_id)
            if spec is None:
                continue

            # 1. Interpolar micro-gaps (<= max_hold_frames) entre observaciones reales
            interp_positions: list[tuple[int, dict[str, Any]]] = []
            for idx_obs, (pos_i, rec_i) in enumerate(obs_list):
                interp_positions.append((pos_i, rec_i))
                if idx_obs + 1 < len(obs_list):
                    next_i, rec_next = obs_list[idx_obs + 1]
                    gap = next_i - pos_i - 1
                    if 1 <= gap <= self.max_hold_frames:
                        for g in range(1, gap + 1):
                            t = g / (gap + 1)
                            ix = (1.0 - t) * rec_i["raw_x_px"] + t * rec_next["raw_x_px"]
                            iy = (1.0 - t) * rec_i["raw_y_px"] + t * rec_next["raw_y_px"]
                            ic = (1.0 - t) * rec_i["confidence"] + t * rec_next["confidence"]
                            held_rec = {
                                "keypoint_idx": spec.model_idx,
                                "roboflow_id": spec.roboflow_id,
                                "keypoint_name": spec.name,
                                "confidence": float(ic),
                                "det_x_px": float(ix),
                                "det_y_px": float(iy),
                                "raw_x_px": float(ix),
                                "raw_y_px": float(iy),
                                "real_x_m": spec.x_m,
                                "real_y_m": spec.y_m,
                                "proj_x_m": None,
                                "proj_y_m": None,
                                "real_x_px": None,
                                "real_y_px": None,
                                "dist_m": None,
                                "is_inlier": False,
                                "is_outlier": False,
                                "color_bgr": spec.color_bgr,
                                "source": "interp_double_pass",
                            }
                            frames_data[pos_i + g]["active_records"].append(held_rec)
                            interp_positions.append((pos_i + g, held_rec))

            # 2. Aplicar EMA bidireccional sobre tramos contiguos del keypoint
            k_len = len(interp_positions)
            sub_start = 0
            while sub_start < k_len:
                sub_end = sub_start + 1
                while sub_end < k_len and interp_positions[sub_end][0] == interp_positions[sub_end - 1][0] + 1:
                    sub_end += 1

                xs = np.array([interp_positions[p][1]["raw_x_px"] for p in range(sub_start, sub_end)], dtype=np.float64)
                ys = np.array([interp_positions[p][1]["raw_y_px"] for p in range(sub_start, sub_end)], dtype=np.float64)
                xs_smooth = _bidirectional_ema_1d(xs, alpha=self.kpt_ema_alpha)
                ys_smooth = _bidirectional_ema_1d(ys, alpha=self.kpt_ema_alpha)

                for offset, p in enumerate(range(sub_start, sub_end)):
                    rec = interp_positions[p][1]
                    rec["det_x_px"] = float(xs_smooth[offset])
                    rec["det_y_px"] = float(ys_smooth[offset])

                sub_start = sub_end
