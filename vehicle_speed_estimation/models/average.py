import numpy as np

from simple_object_detection.typing import FloatVector2D
from simple_object_tracking.datastructures import TrackedObject

from vehicle_speed_estimation.estimation_model import EstimationModel


class TwoPositionsSpeedAverage(EstimationModel):
    """Modelo de estimación de velocidad basado en la media.

    Se introduce el seguimiento de un objeto en la secuencia, y se indica los índices de inicio y
    fin, y se realiza el cálculo medio entre esas dos posiciones.

    Algoritmo
    =========
    Se obtiene el vector de distancia entre la posición inicial y la final, se calcula la cantidad
    de frames transcurridos entre ambos instantes. Por último, se realiza la conversión a km/h con
    los factores introducidos al inicializar la clase.

    El método calculate_speed_kmh(...) devuelve el vector de velocidad medio entre los instantes
    inicial y final.
    """

    def calculate_distance(self,
                           tracked_object: TrackedObject,
                           index_initial: int = 0,
                           index_final: int = -1) -> FloatVector2D:
        """Calcular la distancia entre dos detecciones del objeto.

        :param tracked_object: objeto seguido.
        :param index_initial: índice inicial del historial del objeto.
        :param index_final: índice final del historial del objeto.
        :return: distancia entre las dos posiciones del objeto.
        """
        initial_position = self.get_object_point(tracked_object[index_initial].object)
        final_position = self.get_object_point(tracked_object[index_final].object)
        distance_pixels = tuple(abs(np.array(initial_position) - np.array(final_position)))
        distance_pixels = FloatVector2D(*distance_pixels)
        return self.convert_distance_vector_from_pixels(distance_pixels)

    def calculate_time(self,
                       tracked_object: TrackedObject,
                       index_initial: int = 0,
                       index_final: int = -1):
        """Calcula el tiempo transcurrido entre las dos detecciones del objeto.

        :param tracked_object: objeto seguido.
        :param index_initial: índice inicial del historial del objeto.
        :param index_final: índice final del historial del objeto.
        :return: tiempo transcurrido.
        """
        initial_frame = tracked_object[index_initial].frame
        final_frame = tracked_object[index_final].frame
        time_frames = final_frame - initial_frame
        return self.convert_time_from_frames(time_frames)

    def calculate_speed(self,
                        tracked_object: TrackedObject,
                        index_initial: int = 0,
                        index_final: int = -1):
        """Calcula la velocidad media entre las dos detecciones introducidas por parámetro.

        :param tracked_object: objeto seguido.
        :param index_initial: índice inicial del historial del objeto.
        :param index_final: índice final del historial del objeto.
        :return: velocidad media.
        """
        distance = self.calculate_distance(tracked_object, index_initial, index_final)
        time = self.calculate_time(tracked_object, index_initial, index_final)
        speed = distance.x / time, distance.y / time
        return speed
