from abc import ABC, abstractmethod
from typing import Tuple, List
from enum import Enum

from simple_object_detection.object import Object
from simple_object_detection.typing import Point2D
from simple_object_tracking.typing import ObjectHistory

from vehicle_speed_estimation.utils.conversion import (METERS_TO_KILOMETERS,
                                                       METERS_PER_SECOND_TO_KILOMETERS_PER_HOUR)


class EstimationModel(ABC):
    """
    """

    class ObjectPoint(Enum):
        CENTROID = 0
        TOP_LEFT_CORNER = 1
        TOP_RIGHT_CORNER = 2
        BOTTOM_RIGHT_CORNER = 3
        BOTTOM_LEFT_CORNER = 4

    def __init__(self,
                 pixel_x_to_meters: float,
                 pixel_y_to_meters: float,
                 seconds_per_frame: float,
                 point_selection: ObjectPoint = ObjectPoint.CENTROID):
        """

        :param pixel_x_to_meters: factor de conversión de un píxel en el eje x a cuántos metros
        corresponde.
        :param pixel_y_to_meters: factor de conversión de un píxel en el eje y a cuántos metros
        corresponde.
        :param seconds_per_frame: tiempo (s) que dura cada frame.
        :param point_selection: elegir el punto que se utilizará para realizar los cálculos de la
        posición del vehículo.
        """
        self.pixel_x_to_meters = pixel_x_to_meters
        self.pixel_y_to_meters = pixel_y_to_meters
        self.seconds_per_frame = seconds_per_frame
        self.point_selection = point_selection

    def get_object_point(self, object_detection: Object) -> Point2D:
        points = {
            self.ObjectPoint.CENTROID: object_detection.center,
            self.ObjectPoint.TOP_LEFT_CORNER: object_detection.bounding_box[0],
            self.ObjectPoint.TOP_RIGHT_CORNER: object_detection.bounding_box[1],
            self.ObjectPoint.BOTTOM_RIGHT_CORNER: object_detection.bounding_box[2],
            self.ObjectPoint.BOTTOM_LEFT_CORNER: object_detection.bounding_box[3]
        }
        return points[self.point_selection]

    def distance_vector_px_to_km(self,
                                 distance_px_vector: Tuple[float, float]) -> Tuple[float, float]:
        distance_vector_km = (distance_px_vector[0] * self.pixel_x_to_meters / METERS_TO_KILOMETERS,
                              distance_px_vector[1] * self.pixel_y_to_meters / METERS_TO_KILOMETERS)
        return distance_vector_km

    def speed_vector_px_frame_to_kmh(self, speed_px_f: Tuple[float, float]) -> Tuple[float, float]:
        # Convertir a m/s primeramente.
        speed_vector_m_s = (speed_px_f[0] * self.pixel_x_to_meters / self.seconds_per_frame,
                            speed_px_f[1] * self.pixel_y_to_meters / self.seconds_per_frame)
        speed_vector_kmh = (speed_vector_m_s[0] * METERS_PER_SECOND_TO_KILOMETERS_PER_HOUR,
                            speed_vector_m_s[1] * METERS_PER_SECOND_TO_KILOMETERS_PER_HOUR)
        return speed_vector_kmh
    #
    # @abstractmethod
    # def calculate_distance(self, object_history: ObjectHistory, **kwargs) -> Tuple[float, float]:
    #     """Calcula la distancia en píxeles.
    #
    #     Este método debe ser implementado.
    #
    #     :param object_history: historial del objeto [(frame, detección), ...]
    #     :param kwargs:
    #     :return: vector de distancia calculada en píxeles.
    #     """
    #
    # @abstractmethod
    # def calculate_time(self, object_history: ObjectHistory, **kwargs) -> float:
    #     """Calcula el tiempo transcurrido en frames, es decir, la cantidad de frames transcurridos.
    #
    #     Este método debe ser implementado.
    #
    #     :param object_history: historial del objeto [(frame, detección), ...]
    #     :param kwargs:
    #     :return: cantidad de frames transcurridos.
    #     """
    #
    # @abstractmethod
    # def calculate_speed(self, object_history: ObjectHistory, **kwargs) -> Tuple[float, float]:
    #     """Calcula la velocidad en píxeles/frames.
    #
    #     Este método debe ser implementado.
    #
    #     :param object_history: historial del objeto [(frame, detección), ...]
    #     :param kwargs:
    #     :return: vector de velocidad en píxeles/frames.
    #     """
    #

