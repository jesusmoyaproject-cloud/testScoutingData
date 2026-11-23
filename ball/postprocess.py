import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import supervision as sv
from scipy.interpolate import PchipInterpolator


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
    """Filtro de Kalman 2D con estado [cx, cy, vx, vy] para gating."""

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
        Q = np.diag([4.0 * dt, 4.0 * dt, 12.0 * dt, 12.0 * dt])
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


def _load_and_dedup(csv_path: Path, ball_class_id: int, max_frames: int | None, nms_dist_px: float):
    by_frame: dict[int, list[Detection]] = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["class_id"]) != ball_class_id:
                continue
            frame = int(row["frame"])
            if max_frames is not None and frame >= max_frames:
                continue
            by_frame.setdefault(frame, []).append(
                Detection(
                    frame=frame,
                    class_id=ball_class_id,
                    confidence=float(row["confidence"]),
                    cx=float(row["cx"]),
                    cy=float(row["cy"]),
                    width=float(row["width"]),
                    height=float(row["height"]),
                )
            )

    deduped: dict[int, list[Detection]] = {}
    removed_dups = 0
    for frame, dets in by_frame.items():
        kept: list[Detection] = []
        for d in sorted(dets, key=lambda x: x.confidence, reverse=True):
            if any(np.hypot(d.cx - k.cx, d.cy - k.cy) < nms_dist_px for k in kept):
                removed_dups += 1
                continue
            kept.append(d)
        deduped[frame] = kept
    return deduped, removed_dups


def _build_tracklets(by_frame: dict[int, list[Detection]], pp: dict[str, Any]) -> list[list[Detection]]:
    high_conf = float(pp.get("high_conf", 0.45))
    max_link_gap = int(pp.get("max_link_gap", 4))
    base_gate = float(pp.get("base_gate_px", 25.0))
    max_speed = float(pp.get("max_speed_px", 25.0))

    active: list[tuple[list[Detection], KalmanBall2D]] = []
    finished: list[list[Detection]] = []

    for frame in sorted(by_frame.keys()):
        still_active = []
        for dets, kf in active:
            if frame - dets[-1].frame > max_link_gap:
                finished.append(dets)
            else:
                still_active.append((dets, kf))
        active = still_active

        high_dets = [d for d in by_frame[frame] if d.confidence >= high_conf]
        used = set()

        for dets, kf in active:
            dt = frame - dets[-1].frame
            probe = KalmanBall2D(kf.x[0], kf.x[1])
            probe.x = kf.x.copy()
            probe.P = kf.P.copy()
            pred_x, _ = probe.predict(dt=dt)

            best_i, best_dist = None, float("inf")
            max_euclid = base_gate + max_speed * dt
            for i, d in enumerate(high_dets):
                if i in used:
                    continue
                euclid = float(np.hypot(d.cx - pred_x[0], d.cy - pred_x[1]))
                if euclid <= max_euclid and probe.mahalanobis_sq(d.cx, d.cy) <= 25.0 and euclid < best_dist:
                    best_dist = euclid
                    best_i = i

            if best_i is not None:
                chosen = high_dets[best_i]
                used.add(best_i)
                kf.predict(dt=dt)
                kf.update(chosen.cx, chosen.cy)
                dets.append(chosen)

        for i, d in enumerate(high_dets):
            if i not in used:
                active.append(([d], KalmanBall2D(d.cx, d.cy)))

    finished.extend(dets for dets, _ in active)
    return finished


def _is_static(track: list[Detection], min_frames: int, max_disp: float) -> bool:
    if len(track) < min_frames:
        return False
    xs = [d.cx for d in track]
    ys = [d.cy for d in track]
    return float(np.hypot(max(xs) - min(xs), max(ys) - min(ys))) < max_disp


def _select_global_chain(tracklets: list[list[Detection]], pp: dict[str, Any]) -> list[list[Detection]]:
    s_min = int(pp.get("static_min_frames", 15))
    s_disp = float(pp.get("static_max_disp_px", 12.0))
    max_gap = int(pp.get("max_interp_gap", 75))
    base_gate = float(pp.get("base_gate_px", 25.0))
    max_speed = float(pp.get("max_speed_px", 25.0))

    valid = [t for t in tracklets if len(t) >= 2 and not _is_static(t, s_min, s_disp)]
    if not valid:
        valid = [t for t in tracklets if not _is_static(t, s_min, s_disp)]
    if not valid:
        return []

    valid.sort(key=lambda t: (t[0].frame, -len(t)))
    n = len(valid)
    score = lambda t: sum(d.confidence for d in t) + 0.5 * len(t)
    dp = [score(t) for t in valid]
    parent = [-1] * n

    for i in range(n):
        t_curr = valid[i]
        for j in range(i):
            t_prev = valid[j]
            gap = t_curr[0].frame - t_prev[-1].frame
            if gap <= 0:
                continue
            if gap <= max_gap:
                dist = float(np.hypot(t_curr[0].cx - t_prev[-1].cx, t_curr[0].cy - t_prev[-1].cy))
                if dist > base_gate + max_speed * gap:
                    continue
                penalty = 0.15 * dist
            else:
                penalty = 15.0

            cand = dp[j] + score(t_curr) - penalty
            if cand > dp[i]:
                dp[i] = cand
                parent[i] = j

    idx = int(np.argmax(dp))
    chain = []
    while idx != -1:
        chain.append(valid[idx])
        idx = parent[idx]
    chain.reverse()
    return chain


def _rescue_and_clean(
    chain: list[list[Detection]],
    by_frame: dict[int, list[Detection]],
    pp: dict[str, Any],
) -> tuple[dict[int, Detection], int]:
    low_conf = float(pp.get("low_conf", 0.20))
    max_gap = int(pp.get("max_interp_gap", 75))
    base_gate = float(pp.get("base_gate_px", 25.0))

    selected = {d.frame: d for t in chain for d in t}
    if not selected:
        return selected, 0

    frames_obs = sorted(selected.keys())
    for idx in range(len(frames_obs) - 1):
        f1, f2 = frames_obs[idx], frames_obs[idx + 1]
        gap = f2 - f1
        if gap <= 1 or gap > max_gap:
            continue
        d1, d2 = selected[f1], selected[f2]
        gate = base_gate + 2.0 * min(gap, 15)
        for f in range(f1 + 1, f2):
            if f not in by_frame:
                continue
            alpha = (f - f1) / gap
            ex, ey = d1.cx + alpha * (d2.cx - d1.cx), d1.cy + alpha * (d2.cy - d1.cy)
            best_c, best_d = None, gate
            for c in by_frame[f]:
                if c.confidence < low_conf:
                    continue
                dist = float(np.hypot(c.cx - ex, c.cy - ey))
                if dist < best_d:
                    best_d, best_c = dist, c
            if best_c is not None:
                best_c.source = "rescued_low_conf"
                selected[f] = best_c

    # Eliminar desvíos de ida y vuelta
    removed = 0
    changed = True
    while changed:
        changed = False
        frames = sorted(selected.keys())
        for i in range(1, len(frames) - 1):
            f_prev, f_start = frames[i - 1], frames[i]
            d_prev, d_start = selected[f_prev], selected[f_start]
            if np.hypot(d_start.cx - d_prev.cx, d_start.cy - d_prev.cy) <= 50.0:
                continue
            for k in range(1, 13):
                if i + k >= len(frames):
                    break
                f_after = frames[i + k]
                if f_after - f_prev > max_gap:
                    break
                d_after, d_end = selected[f_after], selected[frames[i + k - 1]]
                jump_back = np.hypot(d_after.cx - d_end.cx, d_after.cy - d_end.cy)
                base_span = np.hypot(d_after.cx - d_prev.cx, d_after.cy - d_prev.cy)
                if jump_back > 50.0 and base_span < 50.0 + 2.5 * (f_after - f_prev):
                    for rm in range(i, i + k):
                        selected.pop(frames[rm], None)
                        removed += 1
                    changed = True
                    break
            if changed:
                break

    return selected, removed


def _interpolate_pchip(
    selected: dict[int, Detection],
    total_frames: int,
    ball_class_id: int,
    pp: dict[str, Any],
) -> dict[int, Detection]:
    if not selected:
        return {}
    max_gap = int(pp.get("max_interp_gap", 75))
    max_tail = int(pp.get("max_tail_extrap", 5))

    obs_f = np.array(sorted(selected.keys()), dtype=np.float64)
    obs_x = np.array([selected[int(f)].cx for f in obs_f], dtype=np.float64)
    obs_y = np.array([selected[int(f)].cy for f in obs_f], dtype=np.float64)
    obs_w = np.array([selected[int(f)].width for f in obs_f], dtype=np.float64)
    obs_h = np.array([selected[int(f)].height for f in obs_f], dtype=np.float64)

    result = dict(selected)
    if len(obs_f) >= 2:
        px, py = PchipInterpolator(obs_f, obs_x), PchipInterpolator(obs_f, obs_y)
        pw, ph = PchipInterpolator(obs_f, obs_w), PchipInterpolator(obs_f, obs_h)
        for i in range(len(obs_f) - 1):
            f1, f2 = int(obs_f[i]), int(obs_f[i + 1])
            if 1 < f2 - f1 <= max_gap:
                for f in range(f1 + 1, f2):
                    result[f] = Detection(
                        frame=f,
                        class_id=ball_class_id,
                        confidence=0.0,
                        cx=float(px(f)),
                        cy=float(py(f)),
                        width=float(max(4.0, pw(f))),
                        height=float(max(4.0, ph(f))),
                        source="interpolated",
                    )

        last_f = int(obs_f[-1])
        if last_f < total_frames - 1:
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
                result[f] = Detection(f, ball_class_id, 0.0, cx, cy, w, h, "extrapolated")

    return result


def run_ball_postprocess(cfg: dict[str, Any], csv_input: Path) -> dict[str, Any]:
    """Limpia falsos positivos, selecciona 1 balón/frame e interpola huecos con PCHIP."""
    out_dir = Path(cfg["task_output_dir"])
    video_path = cfg["video"]
    inf_cfg = cfg["inference"]
    ball_cfg = cfg.get("ball", {})
    pp = ball_cfg.get("postprocess", {})

    ball_class_id = int(ball_cfg.get("class_id", 0))
    max_frames = inf_cfg.get("max_frames")
    video_info = sv.VideoInfo.from_video_path(video_path)
    total_frames = min(video_info.total_frames, max_frames) if (max_frames and video_info.total_frames) else (max_frames or video_info.total_frames)

    by_frame, removed_dups = _load_and_dedup(csv_input, ball_class_id, max_frames, float(pp.get("nms_dist_px", 20.0)))
    tracklets = _build_tracklets(by_frame, pp)
    chain = _select_global_chain(tracklets, pp)
    selected, detours_removed = _rescue_and_clean(chain, by_frame, pp)
    final_dets = _interpolate_pchip(selected, total_frames, ball_class_id, pp)

    csv_clean = out_dir / "detections_clean.csv"
    with open(csv_clean, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "source", "confidence", "x1", "y1", "x2", "y2", "cx", "cy", "width", "height"])
        for frame in sorted(final_dets.keys()):
            d = final_dets[frame]
            x1, y1, x2, y2 = d.xyxy
            writer.writerow([
                d.frame, d.source, f"{d.confidence:.4f}",
                f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                f"{d.cx:.2f}", f"{d.cy:.2f}", f"{d.width:.2f}", f"{d.height:.2f}",
            ])

    counts = {"observed": 0, "rescued_low_conf": 0, "interpolated": 0, "extrapolated": 0}
    for d in final_dets.values():
        counts[d.source] = counts.get(d.source, 0) + 1

    if bool(pp.get("render_video", False)):
        profile_name = cfg.get("name", "run")
        video_clean = Path(cfg.get("videos_output_dir", out_dir)) / f"ball_{profile_name}_clean.mp4"
        frames_gen = sv.get_video_frames_generator(video_path)
        start_t = time.perf_counter()
        with sv.VideoSink(str(video_clean), video_info=video_info) as sink:
            for frame_idx, frame in enumerate(frames_gen):
                if max_frames is not None and frame_idx >= max_frames:
                    break
                ann = frame.copy()
                cv2.rectangle(ann, (12, 12), (210, 48), (0, 0, 0), -1)
                cv2.putText(ann, f"Frame: {frame_idx}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                d = final_dets.get(frame_idx)
                if d is not None:
                    x1, y1, x2, y2 = [int(round(v)) for v in d.xyxy]
                    color = (0, 255, 0) if d.source in ("observed", "rescued_low_conf") else (0, 220, 255)
                    cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
                    lbl = d.source if d.confidence == 0 else f"{d.source} {d.confidence:.2f}"
                    cv2.putText(ann, lbl, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                sink.write_frame(ann)
                if (frame_idx + 1) % 200 == 0:
                    fps = (frame_idx + 1) / max(1e-6, time.perf_counter() - start_t)
                    print(f"[BALL-RENDER] {frame_idx + 1}/{total_frames} | {fps:.1f} FPS", flush=True)

    return {
        "csv_clean": str(csv_clean),
        "frames_with_ball_clean": len(final_dets),
        "coverage_pct": round((len(final_dets) / total_frames) * 100, 2) if total_frames else 0.0,
        "removed_dups": removed_dups,
        "detours_removed": detours_removed,
        "sources": counts,
    }
