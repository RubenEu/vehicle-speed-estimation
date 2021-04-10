import numpy as np
import cv2
import copy
from typing import Tuple, List

from simple_object_detection.typing import Point2D
from simple_object_tracking.typing import SequenceInformation
from simple_object_tracking.datastructures import SequenceObjects


def apply_homography_point2d(point: Point2D, h: np.ndarray) -> Point2D:
    """Aplica una homografía a un punto 2D.

    :param point: punto 2D.
    :param h: matriz de homografía.
    :return: punto 2D en el plano de la homografía.
    """
    x, y, k, = h @ (point[0], point[1], 1)
    return int(x / k), int(y / k)


def apply_homography_sequence(sequence_with_information: SequenceInformation,
                              h: np.ndarray) -> SequenceInformation:
    """Aplicar la homografía a la secuencia.

    :param sequence_with_information: información de la secuencia y la propia secuencia. Tupla
    (width, height, fps, sequence, timestamps).
    :param h: matriz de homografía.
    :return: tupla de la secuencia e información con la homografía aplicada
    (width, height, fps, sequence, timestamps).
    """
    width, height, fps, sequence, timestamps = sequence_with_information
    # Realizar una copia de la secuencia para no editar la original.
    sequence_warped = [frame.copy() for frame in sequence]
    # Aplicar la homografía a cada frame.
    for frame_id, frame in enumerate(sequence_warped):
        sequence_warped[frame_id] = cv2.warpPerspective(frame, h, (width, height))
    return width, height, fps, sequence_warped, timestamps


def apply_homography_objects(objects: SequenceObjects, h: np.ndarray) -> SequenceObjects:
    """Aplicar la homografía al seguimiento de los objetos.

    :param objects: secuencia de objetos (seguimiento).
    :param h: matriz de homografía.
    :return: secuencia de objetos con la homografía aplicada.
    """
    objects = copy.deepcopy(objects)
    # Iterar sobre todos los objetos guardados.
    for obj_uid in range(len(objects)):
        # Aplicar a cada uno de sus registros.
        for frame_seen, object_detected in objects.object_uid(obj_uid):
            # Homografía al centro.
            object_detected.center = apply_homography_point2d(object_detected.center, h)
            # Homografía a las cajas delimitadoras.
            object_detected.bounding_box = tuple(
                apply_homography_point2d(point, h) for point in object_detected.bounding_box)
    return objects
