"""Utilidades geométricas para homografía, refinamiento de inliers y cálculo de distancias en metros."""

import cv2
import numpy as np


def _coordinate_basis(points: np.ndarray) -> np.ndarray:
    """Normaliza coordenadas centrando y escalando para verificar rango numérico."""
    center = points.mean(axis=0, dtype=np.float64)
    scale = float(np.abs(points - center).max()) or 1.0
    basis = np.eye(3, dtype=np.float64)
    basis[:2, :2] *= scale
    basis[:2, 2] = center
    return basis


def _is_valid_homography(matrix: np.ndarray, src: np.ndarray, dst: np.ndarray) -> bool:
    """Verifica rango numérico y consistencia básica de la homografía."""
    try:
        normalized = np.linalg.solve(_coordinate_basis(dst), matrix @ _coordinate_basis(src))
        if np.linalg.matrix_rank(normalized) < 3:
            return False
        det = float(np.linalg.det(matrix))
        if not np.isfinite(det) or abs(det) < 1e-12:
            return False
    except np.linalg.LinAlgError:
        return False
    return True


def compute_homography(
    source_px: np.ndarray,
    target_m: np.ndarray,
    ransac_thresh_m: float = 2.0,
    refine_inliers: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Calcula la matriz de homografía 3x3 de píxeles a metros en cancha.

    Usa RANSAC para descartar outliers y refina por mínimos cuadrados sobre los inliers.
    Retorna `(H, inlier_mask)` o `(None, None)` si no hay suficientes puntos o es degenerada.
    """
    if len(source_px) < 4 or source_px.shape != target_m.shape:
        return None, None

    src = source_px.astype(np.float32)
    dst = target_m.astype(np.float32)

    matrix, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh_m)
    if matrix is None:
        return None, None

    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(src), dtype=bool)
    if int(inliers.sum()) < 4:
        return None, None

    if refine_inliers and int(inliers.sum()) >= 4:
        refined, _ = cv2.findHomography(src[inliers], dst[inliers], method=0)
        if refined is not None and _is_valid_homography(refined, src[inliers], dst[inliers]):
            matrix = refined

    if not _is_valid_homography(matrix, src[inliers], dst[inliers]):
        return None, None

    return matrix, inliers


def project_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Proyecta puntos 2D `(N, 2)` usando una matriz de homografía 3x3."""
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    reshaped = points.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.perspectiveTransform(reshaped, matrix)
    return transformed.reshape(-1, 2).astype(np.float32)


def reproject_real_to_px(matrix_px_to_m: np.ndarray, real_m: np.ndarray) -> np.ndarray | None:
    """Proyecta coordenadas reales de la cancha (en metros) de vuelta a píxeles usando H^-1."""
    if len(real_m) == 0:
        return np.empty((0, 2), dtype=np.float32)
    try:
        inv_h = np.linalg.inv(matrix_px_to_m)
    except np.linalg.LinAlgError:
        return None
    return project_points(inv_h, real_m)


def compute_distances_m(proj_m: np.ndarray, real_m: np.ndarray) -> np.ndarray:
    """Calcula la distancia euclidiana en metros entre posición proyectada y real."""
    if len(proj_m) == 0:
        return np.empty((0,), dtype=np.float32)
    return np.linalg.norm(proj_m - real_m, axis=1).astype(np.float32)
