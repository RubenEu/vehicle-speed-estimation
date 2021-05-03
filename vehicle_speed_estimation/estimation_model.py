from abc import ABC
from enum import Enum

from simple_object_detection.object import Object
from simple_object_detection.typing import Point2D, FloatVector2D


class EstimationModel(ABC):
    """Modelo abstracto para la implementación de modelos de estimación de velocidad.
    """

    class ObjectPoint(Enum):
        CENTROID = 0
        TOP_LEFT_CORNER = 1
        TOP_RIGHT_CORNER = 2
        BOTTOM_RIGHT_CORNER = 3
        BOTTOM_LEFT_CORNER = 4

    class TimeUnit(Enum):
        FRAME = 0
        SECOND = 1
        HOUR = 2

    class LengthUnit(Enum):
        PIXEL = 0
        METER = 1
        KILOMETER = 2

    def __init__(self,
                 pixel_to_meters: FloatVector2D,
                 frames_per_second: float,
                 object_point: ObjectPoint = ObjectPoint.CENTROID,
                 time_unit: TimeUnit = TimeUnit.HOUR,
                 length_unit: LengthUnit = LengthUnit.KILOMETER):
        """

        :param pixel_to_meters: vector de factor de conversión de un píxel a metros.
        :param frames_per_second: frames que transcurren por cada segundo.
        :param object_point: punto que se utilizará para realizar los cálculos de la posición del
        objeto.
        :param time_unit: unidad para la medida del tiempo.
        :param length_unit: unidad para la medida del espacio.
        """
        self.pixel_to_meters = pixel_to_meters
        self.frames_per_second = frames_per_second
        self.object_point = object_point
        self.time_unit = time_unit
        self.length_unit = length_unit

    def convert_time_from_frames(self, frames: int) -> float:
        """Convierte el tiempo desde frames a la unidad especificada al instanciar la clase.

        :param frames: cantidad de tiempo en frames.
        :return: cantidad de tiempo en la unidad especificada al instancia la clase.
        """
        if self.time_unit == self.TimeUnit.SECOND:
            return frames / self.frames_per_second
        elif self.time_unit == self.TimeUnit.HOURS:
            return frames / self.frames_per_second / 3600
        return frames

    def convert_distance_vector_from_pixels(self, distance_vector: FloatVector2D) -> FloatVector2D:
        """Convierte el vector de distancia desde píxeles a la unidad especificada al instanciar la
        clase.

        :param distance_vector: vector de distancia en píxeles.
        :return: cantidad de espacio en la unidad especificada al instanciar la clase.
        """
        if self.length_unit == self.LengthUnit.METER:
            distance_x = distance_vector.x * self.pixel_to_meters.x
            distance_y = distance_vector.y * self.pixel_to_meters.y
            return FloatVector2D(distance_x, distance_y)
        elif self.length_unit == self.LengthUnit.KILOMETER:
            distance_x = distance_vector.x * self.pixel_to_meters.x / 1000
            distance_y = distance_vector.y * self.pixel_to_meters.y / 1000
            return FloatVector2D(distance_x, distance_y)
        return distance_vector

    def get_object_point(self, object_detection: Object) -> Point2D:
        """Obtiene el punto del objeto especificado al instanciar la clase.

        :param object_detection: detección del objeto.
        :return: punto del objeto.
        """
        points = {
            self.ObjectPoint.CENTROID: object_detection.center,
            self.ObjectPoint.TOP_LEFT_CORNER: object_detection.bounding_box[0],
            self.ObjectPoint.TOP_RIGHT_CORNER: object_detection.bounding_box[1],
            self.ObjectPoint.BOTTOM_RIGHT_CORNER: object_detection.bounding_box[2],
            self.ObjectPoint.BOTTOM_LEFT_CORNER: object_detection.bounding_box[3]
        }
        return points[self.object_point]
