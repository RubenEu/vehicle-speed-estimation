import cv2
from typing import Tuple, List

from simple_object_detection.typing import Image, Point2D
from simple_object_detection.utils import draw_bounding_boxes
from simple_object_tracking.typing import ObjectHistory


def draw_object_tracking(image: Image, positions: List[Point2D]) -> Image:
    """Dibuja el seguimiento del objeto

    TODO: Añadir colores por parámetro.

    :param image: imagen sobre la que dibujar el seguimiento.
    :param positions: posiciones del seguimiento del objeto. Bien sea el centro, una esquina, etc.
    :return: imagen con el seguimiento dibujado.
    """
    image = image.copy()
    # Dibujar cada una de las posiciones
    prev_position = positions[0]
    for position in positions:
        cv2.line(image, position, prev_position, (255, 43, 155), 2, cv2.LINE_AA)
        cv2.circle(image, position, 0, (107, 37, 74), 5, cv2.LINE_AA)
        prev_position = position
    return image


def draw_tracking_and_bounding_box(image: Image, object_history: ObjectHistory) -> Image:
    # Evitar editar la imagen pasada por referencia.
    image = image.copy()
    # Obtener las posiciones
    positions_center = [obj.center for _, obj in object_history]
    positions_box_bottom_left = [obj.bounding_box[3] for _, obj in object_history]
    positions_box_bottom_right = [obj.bounding_box[2] for _, obj in object_history]
    positions_box_top_right = [obj.bounding_box[1] for _, obj in object_history]
    positions_box_top_left = [obj.bounding_box[0] for _, obj in object_history]
    # Agregar los puntos del centro y de las dos esquinas inferiores.
    image = draw_object_tracking(image, positions_center)
    image = draw_object_tracking(image, positions_box_bottom_left)
    image = draw_object_tracking(image, positions_box_bottom_right)
    image = draw_object_tracking(image, positions_box_top_right)
    image = draw_object_tracking(image, positions_box_top_left)
    # Agregar la bounding box del inicio y del final.
    image = draw_bounding_boxes(image, [object_history[0][1]])
    image = draw_bounding_boxes(image, [object_history[-1][1]])
    return image
