from typing import Tuple
import numpy as np

from simple_object_tracking.typing import ObjectHistory
from simple_object_tracking.utils import calculate_euclidean_distance

from vehicle_speed_estimation.estimation_model import EstimationModel


class TwoPositionsSpeedAverage(EstimationModel):
    """Modelo de estimación de velocidad basado en la media.

    Se introduce el historial del objeto de detecciones en la secuencia, y se indica los índices de
    inicio y fin, y se realiza el cálculo medio entre esas dos posiciones.

    Se obtiene el vector de distancia entre la posición inicial y la final, se calcula la cantidad
    de frames transcurridos entre ambos instantes. Por último, se realiza la conversión a km/h con
    los factores introducidos al inicializar la clase.

    El método calculate_speed_kmh(...) devuelve el vector de velocidad medio entre los instantes
    inicial y final.
    """

    def calculate_distance_px(self, object_history: ObjectHistory, initial_index: int = 0,
                              final_index: int = -1) -> Tuple[float, float]:
        """Calcular la distancia en píxeles entre dos detecciones del objeto.

        :param object_history: historial del objeto [(frame, deteccion), ...]
        :param initial_index: índice inicial del historial del objeto.
        :param final_index: índice final del historial del objeto.
        :return: distancia entre las dos posiciones del objeto (en píxeles).
        """
        _, initial_detection = object_history[initial_index]
        _, final_detection = object_history[final_index]
        initial_pos = self.get_object_point(initial_detection)
        final_pos = self.get_object_point(final_detection)
        distance_vector = tuple(abs(np.array(final_pos) - np.array(initial_pos)))
        return distance_vector

    def calculate_time_frames(self, object_history: ObjectHistory, initial_index: int = 0,
                              final_index: int = -1) -> float:
        """Calcula el tiempo transcurrido en frames entre las dos detecciones del objeto.

        :param object_history: historial del objeto [(frame, deteccion), ...]
        :param initial_index: índice inicial del historial del objeto.
        :param final_index: índice final del historial del objeto.
        :return: tiempo transcurrido en frames.
        """
        initial_frame, _ = object_history[initial_index]
        final_frame, _ = object_history[final_index]
        return final_frame - initial_frame

    def calculate_speed_kmh(self, object_history: ObjectHistory, initial_index: int = 0,
                            final_index: int = -1) -> Tuple[float, float]:
        """Calcula la velocidad media entre las dos detecciones.

        Primeramente se calcula en px/frame y se realiza la conversión a km/h con los factores
        introducidos al inicializar el modelo.

        :param object_history: historial del objeto [(frame, deteccion), ...]
        :param initial_index: índice inicial del historial del objeto.
        :param final_index: índice final del historial del objeto.
        :return: velocidad calculada en km/h.
        """
        distance_px = self.calculate_distance_px(object_history, initial_index, final_index)
        time_frames = self.calculate_time_frames(object_history, initial_index, final_index)
        speed_px_frame = distance_px[0] / time_frames, distance_px[1] / time_frames
        speed_km_h = self.speed_vector_px_frame_to_kmh(speed_px_frame)
        return speed_km_h
