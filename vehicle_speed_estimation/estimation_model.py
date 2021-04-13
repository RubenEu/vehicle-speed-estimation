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

    class Units(Enum):
        PIXELS_FRAME = 0
        METERS_SECOND = 1
        KILOMETERS_HOURS = 2

    def __init__(self,
                 pixel_to_meters: Tuple[float, float],
                 seconds_per_frame: float,
                 point_selection: ObjectPoint = ObjectPoint.CENTROID,
                 units: Units = Units.KILOMETERS_HOURS):
        """

        :param pixel_to_meters: vector de factor de conversión de un píxel a metros.
        :param seconds_per_frame: tiempo (s) que dura cada frame.
        :param point_selection: elegir el punto que se utilizará para realizar los cálculos de la
        posición del vehículo.
        :param units: unidades para la medida de la distancia, tiempo y velocidad.
        """
        self.pixel_to_meters = pixel_to_meters
        self.seconds_per_frame = seconds_per_frame
        self.point_selection = point_selection
        self.units = units

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
        distance_km = (distance_px_vector[0] * self.pixel_to_meters[0] / METERS_TO_KILOMETERS,
                       distance_px_vector[1] * self.pixel_to_meters[1] / METERS_TO_KILOMETERS)
        return distance_km

    def speed_vector_px_frame_to_kmh(self, speed_px_f: Tuple[float, float]) -> Tuple[float, float]:
        # Convertir a m/s primeramente.
        speed_vector_m_s = (speed_px_f[0] * self.pixel_to_meters[0] / self.seconds_per_frame,
                            speed_px_f[1] * self.pixel_to_meters[1] / self.seconds_per_frame)
        speed_vector_kmh = (speed_vector_m_s[0] * METERS_PER_SECOND_TO_KILOMETERS_PER_HOUR,
                            speed_vector_m_s[1] * METERS_PER_SECOND_TO_KILOMETERS_PER_HOUR)
        return speed_vector_kmh
