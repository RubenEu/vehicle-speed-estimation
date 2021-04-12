# https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
# https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
from typing import List, Tuple
import numpy as np

from simple_object_detection.typing import Point2D
from simple_object_tracking.utils import calculate_euclidean_distance


def line_pixel_points(p: Point2D, q: Point2D) -> List[Point2D]:
    """Crea una recta definida por todos sus puntos en la imagen dado un punto inicial y un punto
    final.

    :param p: punto p.
    :param q: punto q.
    :return: lista de puntos que definen la recta píxel a píxel.
    """
    distance = calculate_euclidean_distance(p, q)
    line_x: List[int] = list(np.linspace(p[0], q[0]).astype(np.int32))
    line_y: List[int] = list(np.linspace(p[1], q[1]).astype(np.int32))
    line_xy = list(zip(line_x, line_y))
    # Eliminar duplicados manteniendo el orden.
    line_xy = list(dict.fromkeys(line_xy))
    return line_xy


def point_distance_to_line(point: Point2D, line: Tuple[Point2D, Point2D]) -> float:
    """Calcula la distancia entre un punto y una recta definida por sus extremos.

    Extraído de:
    https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points

    :param point: punto (x, y).
    :param line: recta definida por los dos extremos [(x0, y0), (x1, y1)].
    :return: mínima distancia entre el punto y la recta.
    """
    p1, p2 = np.array(line[0]), np.array(line[1])
    p3 = np.array(point)
    return np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)


def find_closest_position_to_line(positions: List[Point2D], line: Tuple[Point2D, Point2D]) -> int:
    """Busca el frame en el que el centro del objeto está más cercano a la
    línea especificada.

    :param positions:
    :param line:
    :return: índice de la posición más cercana.
    """
    distances = [point_distance_to_line(p, line) for p in positions]
    return distances.index(min(distances))
