"""Diccionario de datos y configuración geométrica de la cancha de fútbol.

Basado en `SoccerPitchConfiguration` de Roboflow (`roboflow/sports` / `football-ai.ipynb`),
adaptado al modelo de 30 keypoints (`scoutingDataKeyPointV2.v2`) y con soporte opcional
para modelos de 32 keypoints.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PitchKeypointSpec:
    """Especificación de un keypoint de la cancha."""

    model_idx: int
    roboflow_id: int
    name: str
    x_cm: float
    y_cm: float
    color_bgr: tuple[int, int, int]

    @property
    def x_m(self) -> float:
        return self.x_cm / 100.0

    @property
    def y_m(self) -> float:
        return self.y_cm / 100.0


@dataclass
class SoccerPitchConfiguration:
    """Dimensiones estándar de cancha en centímetros (Roboflow football-ai.ipynb)."""

    width: int = 7000                 # 70.0 m
    length: int = 12000               # 120.0 m
    penalty_box_width: int = 4100     # 41.0 m
    penalty_box_length: int = 2015    # 20.15 m
    goal_box_width: int = 1832        # 18.32 m
    goal_box_length: int = 550        # 5.50 m
    centre_circle_radius: int = 915   # 9.15 m
    penalty_spot_distance: int = 1100 # 11.00 m

    @property
    def vertices(self) -> list[tuple[float, float]]:
        """Los 32 vértices originales de SoccerPitchConfiguration (en cm)."""
        return [
            (0.0, 0.0),  # 1
            (0.0, (self.width - self.penalty_box_width) / 2.0),  # 2
            (0.0, (self.width - self.goal_box_width) / 2.0),  # 3
            (0.0, (self.width + self.goal_box_width) / 2.0),  # 4
            (0.0, (self.width + self.penalty_box_width) / 2.0),  # 5
            (0.0, float(self.width)),  # 6
            (float(self.goal_box_length), (self.width - self.goal_box_width) / 2.0),  # 7
            (float(self.goal_box_length), (self.width + self.goal_box_width) / 2.0),  # 8
            (float(self.penalty_spot_distance), self.width / 2.0),  # 9 (penalty spot izq)
            (float(self.penalty_box_length), (self.width - self.penalty_box_width) / 2.0),  # 10
            (float(self.penalty_box_length), (self.width - self.goal_box_width) / 2.0),  # 11
            (float(self.penalty_box_length), (self.width + self.goal_box_width) / 2.0),  # 12
            (float(self.penalty_box_length), (self.width + self.penalty_box_width) / 2.0),  # 13
            (self.length / 2.0, 0.0),  # 14
            (self.length / 2.0, self.width / 2.0 - self.centre_circle_radius),  # 15
            (self.length / 2.0, self.width / 2.0 + self.centre_circle_radius),  # 16
            (self.length / 2.0, float(self.width)),  # 17
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2.0),  # 18
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2.0),  # 19
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2.0),  # 20
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2.0),  # 21
            (float(self.length - self.penalty_spot_distance), self.width / 2.0),  # 22 (penalty spot der)
            (float(self.length - self.goal_box_length), (self.width - self.goal_box_width) / 2.0),  # 23
            (float(self.length - self.goal_box_length), (self.width + self.goal_box_width) / 2.0),  # 24
            (float(self.length), 0.0),  # 25
            (float(self.length), (self.width - self.penalty_box_width) / 2.0),  # 26
            (float(self.length), (self.width - self.goal_box_width) / 2.0),  # 27
            (float(self.length), (self.width + self.goal_box_width) / 2.0),  # 28
            (float(self.length), (self.width + self.penalty_box_width) / 2.0),  # 29
            (float(self.length), float(self.width)),  # 30
            (self.length / 2.0 - self.centre_circle_radius, self.width / 2.0),  # 31
            (self.length / 2.0 + self.centre_circle_radius, self.width / 2.0),  # 32
        ]

    edges_1based: list[tuple[int, int]] = field(
        default_factory=lambda: [
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
            (7, 8), (10, 11), (11, 12), (12, 13),
            (14, 15), (15, 16), (16, 17),
            (18, 19), (19, 20), (20, 21), (23, 24),
            (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
            (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
        ]
    )


_PINK_BGR = (147, 20, 255)    # Mitad izquierda (#FF1493)
_BLUE_BGR = (255, 191, 0)     # Línea central / círculo (#00BFFF)
_ORANGE_BGR = (71, 99, 255)   # Mitad derecha (#FF6347)

# Nombres descriptivos para los 32 vértices de Roboflow (1-based)
_ROBOFLOW_NAMES: dict[int, tuple[str, tuple[int, int, int]]] = {
    1: ("left_corner_top", _PINK_BGR),
    2: ("left_penalty_box_top_goal_line", _PINK_BGR),
    3: ("left_goal_box_top_goal_line", _PINK_BGR),
    4: ("left_goal_box_bottom_goal_line", _PINK_BGR),
    5: ("left_penalty_box_bottom_goal_line", _PINK_BGR),
    6: ("left_corner_bottom", _PINK_BGR),
    7: ("left_goal_box_top_corner", _PINK_BGR),
    8: ("left_goal_box_bottom_corner", _PINK_BGR),
    9: ("left_penalty_spot", _PINK_BGR),
    10: ("left_penalty_box_top_corner", _PINK_BGR),
    11: ("left_penalty_box_top_goal_proj", _PINK_BGR),
    12: ("left_penalty_box_bottom_goal_proj", _PINK_BGR),
    13: ("left_penalty_box_bottom_corner", _PINK_BGR),
    14: ("halfway_line_top", _BLUE_BGR),
    15: ("center_circle_top", _BLUE_BGR),
    16: ("center_circle_bottom", _BLUE_BGR),
    17: ("halfway_line_bottom", _BLUE_BGR),
    18: ("right_penalty_box_top_corner", _ORANGE_BGR),
    19: ("right_penalty_box_top_goal_proj", _ORANGE_BGR),
    20: ("right_penalty_box_bottom_goal_proj", _ORANGE_BGR),
    21: ("right_penalty_box_bottom_corner", _ORANGE_BGR),
    22: ("right_penalty_spot", _ORANGE_BGR),
    23: ("right_goal_box_top_corner", _ORANGE_BGR),
    24: ("right_goal_box_bottom_corner", _ORANGE_BGR),
    25: ("right_corner_top", _ORANGE_BGR),
    26: ("right_penalty_box_top_goal_line", _ORANGE_BGR),
    27: ("right_goal_box_top_goal_line", _ORANGE_BGR),
    28: ("right_goal_box_bottom_goal_line", _ORANGE_BGR),
    29: ("right_penalty_box_bottom_goal_line", _ORANGE_BGR),
    30: ("right_corner_bottom", _ORANGE_BGR),
    31: ("center_circle_left", _BLUE_BGR),
    32: ("center_circle_right", _BLUE_BGR),
}

# El modelo homography_v3 (scoutingDataKeyPointV2.v2, kpt_shape=[30, 3])
# usa los 30 vértices de Roboflow excluyendo los puntos penales (#9 y #22).
_ROBOFLOW_IDS_30: list[int] = [i for i in range(1, 33) if i not in (9, 22)]


def build_keypoint_dictionary(
    num_keypoints: int = 30,
    config: SoccerPitchConfiguration | None = None,
) -> list[PitchKeypointSpec]:
    """Construye el diccionario ordenado de keypoints indexado por `model_idx`."""
    cfg = config or SoccerPitchConfiguration()
    vertices = cfg.vertices
    if num_keypoints == 30:
        rf_ids = _ROBOFLOW_IDS_30
    elif num_keypoints == 32:
        rf_ids = list(range(1, 33))
    else:
        raise ValueError(f"Número de keypoints no soportado: {num_keypoints} (esperado 30 o 32).")

    specs: list[PitchKeypointSpec] = []
    for model_idx, rf_id in enumerate(rf_ids):
        x_cm, y_cm = vertices[rf_id - 1]
        name, color = _ROBOFLOW_NAMES[rf_id]
        specs.append(
            PitchKeypointSpec(
                model_idx=model_idx,
                roboflow_id=rf_id,
                name=name,
                x_cm=x_cm,
                y_cm=y_cm,
                color_bgr=color,
            )
        )
    return specs


def build_edges_for_model(
    specs: list[PitchKeypointSpec],
    config: SoccerPitchConfiguration | None = None,
) -> list[tuple[int, int]]:
    """Convierte las aristas 1-based de Roboflow a pares de índices `model_idx`."""
    cfg = config or SoccerPitchConfiguration()
    rf_to_model = {spec.roboflow_id: spec.model_idx for spec in specs}
    edges: list[tuple[int, int]] = []
    for a_rf, b_rf in cfg.edges_1based:
        if a_rf in rf_to_model and b_rf in rf_to_model:
            edges.append((rf_to_model[a_rf], rf_to_model[b_rf]))
    return edges
