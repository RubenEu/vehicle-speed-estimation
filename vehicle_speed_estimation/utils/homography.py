import copy
import cv2
import numpy as np

from simple_object_detection.typing import Point2D, Image, BoundingBox
from simple_object_tracking.datastructures import TrackedObjects


def apply_homography_point2d(point: Point2D, h: np.ndarray) -> Point2D:
    """Aplica una homografía a un punto 2D.

    :param point: punto 2D.
    :param h: matriz de homografía.
    :return: punto 2D en el plano de la homografía.
    """
    x, y, k, = h @ (point[0], point[1], 1)
    return Point2D(int(x / k), int(y / k))


def apply_homography_frame(frame: Image, h: np.ndarray) -> Image:
    """Aplica la homografía a un frame.

    :param frame: imagen.
    :param h: matriz de homografía.
    :return: frame con la homografía aplicada.
    """
    height, width, _ = frame.shape
    # Copiar frame para no editarlo.
    frame = frame.copy()
    # Aplicar homografía y devolverlo
    frame_h = cv2.warpPerspective(frame, h, (width, height))
    return frame_h


def apply_homography_objects(tracked_objects: TrackedObjects, h: np.ndarray) -> TrackedObjects:
    """Aplicar la homografía al seguimiento de los objetos.

    :param tracked_objects: secuencia de objetos (seguimiento).
    :param h: matriz de homografía.
    :return: secuencia de objetos con la homografía aplicada.
    """
    tracked_objects = copy.deepcopy(tracked_objects)
    # Iterar sobre todos los objetos seguidos.
    for tracked_object in tracked_objects:
        # Aplicar a cada una de sus detecciones en el seguimiento.
        for object_detection in tracked_object:
            object_ = object_detection.object
            # Homografía al centro.
            object_.center = apply_homography_point2d(object_.center, h)
            # Homografía a la bounding box.
            bounding_box_h = tuple(apply_homography_point2d(p, h) for p in object_.bounding_box)
            object_.bounding_box = BoundingBox(*bounding_box_h)
    return tracked_objects
