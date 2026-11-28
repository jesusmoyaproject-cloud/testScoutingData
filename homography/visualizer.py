"""Renderizado visual de keypoints detectados, keypoint real (H^-1), distancia en metros y minimapa 2D."""

from typing import Sequence
import cv2
import numpy as np
from homography.geometry import project_points
from homography.pitch_config import PitchKeypointSpec, SoccerPitchConfiguration


def _draw_label_box(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    bg_color: tuple[int, int, int] = (20, 20, 20),
    text_color: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.42,
    thickness: int = 1,
) -> None:
    """Dibuja texto legible con fondo oscuro."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    h_img, w_img = img.shape[:2]
    x = max(4, min(org[0], w_img - tw - 8))
    y = max(th + 6, min(org[1], h_img - baseline - 4))
    cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + baseline + 3), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def _draw_real_marker(img: np.ndarray, x: int, y: int, size: int = 7) -> None:
    """Dibuja un rombo con cruz para representar la posición real del keypoint."""
    pts = np.array([[x, y - size], [x + size, y], [x, y + size], [x - size, y]], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 235, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.line(img, (x - 4, y), (x + 4, y), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (x, y - 4), (x, y + 4), (255, 255, 255), 1, cv2.LINE_AA)


def _draw_minimap(
    canvas_w: int,
    canvas_h: int,
    pitch_cfg: SoccerPitchConfiguration,
    specs: Sequence[PitchKeypointSpec],
    edges: Sequence[tuple[int, int]],
    detected_records: list[dict],
) -> np.ndarray:
    """Dibuja un minimapa 2D de la cancha con puntos reales vs proyectados."""
    pad = 12
    mm = np.full((canvas_h, canvas_w, 3), (26, 72, 30), dtype=np.uint8)
    length_m = pitch_cfg.length / 100.0
    width_m = pitch_cfg.width / 100.0
    scale_x = (canvas_w - 2 * pad) / length_m
    scale_y = (canvas_h - 2 * pad) / width_m

    def to_mm(xm: float, ym: float) -> tuple[int, int]:
        return (int(round(pad + xm * scale_x)), int(round(pad + ym * scale_y)))

    line_col = (210, 225, 210)
    for a_idx, b_idx in edges:
        pa = to_mm(specs[a_idx].x_m, specs[a_idx].y_m)
        pb = to_mm(specs[b_idx].x_m, specs[b_idx].y_m)
        cv2.line(mm, pa, pb, line_col, 1, cv2.LINE_AA)

    center_pt = to_mm(length_m / 2.0, width_m / 2.0)
    radius_px = int(round((pitch_cfg.centre_circle_radius / 100.0) * scale_x))
    cv2.circle(mm, center_pt, radius_px, line_col, 1, cv2.LINE_AA)

    for rec in detected_records:
        real_pt = to_mm(rec["real_x_m"], rec["real_y_m"])
        cv2.circle(mm, real_pt, 4, (0, 235, 255), 1, cv2.LINE_AA)
        if rec["proj_x_m"] is not None and rec["proj_y_m"] is not None:
            proj_pt = to_mm(rec["proj_x_m"], rec["proj_y_m"])
            if 0 <= proj_pt[0] < canvas_w and 0 <= proj_pt[1] < canvas_h:
                cv2.line(mm, real_pt, proj_pt, (0, 215, 255), 1, cv2.LINE_AA)
                cv2.circle(mm, proj_pt, 3, rec["color_bgr"], -1, cv2.LINE_AA)

    cv2.rectangle(mm, (0, 0), (canvas_w - 1, canvas_h - 1), (180, 180, 180), 1)
    return mm


def render_homography_frame(
    frame: np.ndarray,
    frame_idx: int,
    detected_records: list[dict],
    homography_matrix: np.ndarray | None,
    pitch_cfg: SoccerPitchConfiguration,
    specs: Sequence[PitchKeypointSpec],
    edges: Sequence[tuple[int, int]],
    draw_wireframe: bool = True,
    draw_minimap: bool = True,
    rejected_count: int = 0,
) -> np.ndarray:
    """Dibuja puntos detectados, keypoint real reproyectado (H^-1), distancia en metros y HUD."""
    out = frame.copy()
    h_img, w_img = out.shape[:2]

    dists = [r["dist_m"] for r in detected_records if r["dist_m"] is not None]
    mean_dist = float(np.mean(dists)) if dists else None
    max_dist = float(np.max(dists)) if dists else None

    # 1. Reproyectar wireframe de la cancha si la homografía es estable
    if draw_wireframe and homography_matrix is not None and (mean_dist is None or mean_dist < 8.0):
        try:
            inv_h = np.linalg.inv(homography_matrix)
            real_pts_m = np.array([[s.x_m, s.y_m] for s in specs], dtype=np.float32)
            reproj_px = project_points(inv_h, real_pts_m)
            for a_idx, b_idx in edges:
                pa = reproj_px[a_idx]
                pb = reproj_px[b_idx]
                if (
                    -w_img < pa[0] < 2 * w_img
                    and -h_img < pa[1] < 2 * h_img
                    and -w_img < pb[0] < 2 * w_img
                    and -h_img < pb[1] < 2 * h_img
                ):
                    cv2.line(
                        out,
                        (int(round(pa[0])), int(round(pa[1]))),
                        (int(round(pb[0])), int(round(pb[1]))),
                        (255, 220, 80),
                        1,
                        cv2.LINE_AA,
                    )
        except np.linalg.LinAlgError:
            pass

    # 2. Dibujar cada keypoint detectado + su keypoint real (H^-1) + etiqueta
    for rec in detected_records:
        px = int(round(rec["det_x_px"]))
        py = int(round(rec["det_y_px"]))
        color = rec["color_bgr"]
        is_held = rec.get("source") == "held"

        # Keypoint real reproyectado desde coordenadas reales de cancha
        rx_px = rec.get("real_x_px")
        ry_px = rec.get("real_y_px")
        if rx_px is not None and ry_px is not None:
            rx_i, ry_i = int(round(rx_px)), int(round(ry_px))
            if -w_img // 2 < rx_i < int(1.5 * w_img) and -h_img // 2 < ry_i < int(1.5 * h_img):
                cv2.line(out, (px, py), (rx_i, ry_i), (0, 235, 255), 1, cv2.LINE_AA)
                _draw_real_marker(out, rx_i, ry_i, size=6)

        # Keypoint detectado
        if is_held:
            cv2.circle(out, (px, py), 6, color, 2, cv2.LINE_AA)
        else:
            cv2.circle(out, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.circle(out, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)

        dist_str = f" | d={rec['dist_m']:.2f}m" if rec["dist_m"] is not None else ""
        label = (
            f"#{rec['keypoint_idx']} px({px},{py}) "
            f"real({rec['real_x_m']:.1f},{rec['real_y_m']:.1f})m{dist_str}"
        )
        _draw_label_box(out, label, (px + 10, py - 8), text_color=(245, 245, 245))

    # 3. HUD superior izquierdo con leyenda
    homo_status = "OK" if homography_matrix is not None else "INSUFICIENTE (<4 pts)"
    mean_str = f"{mean_dist:.2f} m" if mean_dist is not None else "N/A"
    max_str = f"{max_dist:.2f} m" if max_dist is not None else "N/A"
    rej_str = f" | Saltos filtrados: {rejected_count}" if rejected_count > 0 else ""
    hud_text = (
        f"Frame {frame_idx} | Kpts: {len(detected_records)}/{len(specs)}{rej_str} | "
        f"H: {homo_status} | Dist media: {mean_str} | Max: {max_str} | [Circulo=Detectado, Rombo=Real]"
    )
    _draw_label_box(out, hud_text, (12, 26), bg_color=(15, 18, 22), text_color=(80, 255, 180), font_scale=0.50)

    # 4. Minimapa 2D en esquina inferior derecha
    if draw_minimap and w_img >= 640 and h_img >= 360:
        mm_w = min(320, max(200, w_img // 4))
        mm_h = int(round(mm_w * (pitch_cfg.width / pitch_cfg.length)))
        mm = _draw_minimap(mm_w, mm_h, pitch_cfg, specs, edges, detected_records)
        x0 = w_img - mm_w - 14
        y0 = h_img - mm_h - 14
        out[y0 : y0 + mm_h, x0 : x0 + mm_w] = mm

    return out
