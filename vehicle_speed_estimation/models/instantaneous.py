from typing import List, Tuple
import numpy as np

from simple_object_tracking.typing import ObjectHistory
from simple_object_tracking.utils import calculate_euclidean_distance

from vehicle_speed_estimation.estimation_model import EstimationModel
from vehicle_speed_estimation.utils.conversion import (METERS_TO_KILOMETERS,
                                                       METERS_PER_SECOND_TO_KILOMETERS_PER_HOUR)


class InstantaneousSpeed(EstimationModel):
    """

    """

    def calculate_speeds_px_frame(self, object_history: ObjectHistory) -> List[Tuple[float, float]]:
        """

        :param object_history:
        :return:
        """
        # Desempaquetar información.
        frames, detections = zip(*object_history)
        positions = [self.get_object_point(detection) for detection in detections]
        index_prev = 0
        speeds = list()
        for index in range(1, len(positions)):
            # Cálculo de distancia.
            position_prev = np.array(positions[index_prev])
            position = np.array(positions[index])
            distance = position - position_prev
            # Cálculo de tiempo.
            time = frames[index] - frames[index_prev]
            # Cálculo de velocidad.
            speed = distance / time
            # Añadir a la lista.
            speeds.append(tuple(speed))
            # Actualizar la posición previa.
            index_prev = index
        return speeds

    def calculate_speeds_kmh(self, object_history: ObjectHistory) -> List[Tuple[float, float]]:
        speeds_px_frame = self.calculate_speeds_px_frame(object_history)
        speeds_km_h = [self.speed_vector_px_frame_to_kmh(speed) for speed in speeds_px_frame]
        return speeds_km_h
