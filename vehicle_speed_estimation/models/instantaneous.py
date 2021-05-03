from typing import List, Tuple, Callable, Any
from enum import Enum
import numpy as np

from simple_object_detection.typing import FloatVector2D
from simple_object_tracking.datastructures import TrackedObject

from vehicle_speed_estimation.estimation_model import EstimationModel


class InstantaneousVelocity(EstimationModel):
    """Estimación de los vectores de velocidad usando diferenciación discreta.
    """

    def calculate_velocities(self, tracked_object: TrackedObject) -> List[FloatVector2D]:
        """Realiza el cálculo de las velocidades en cada instante que fue detectado el objeto.

        La unidad de medida de las velocidades vendrá determinada por la introducida al instanciar
        el modelo con el parámetro `units`.

        :param tracked_object: seguimiento del objeto.
        :return: lista de las velocidades en cada instante.
        """
        # Desempaquetar información.
        frames, detections = tracked_object.frames, tracked_object.detections
        positions_pixels = [self.get_object_point(detection) for detection in detections]
        # Convertir a las unidades deseadas.
        positions = [self.convert_distance_vector_from_pixels(FloatVector2D(*p))
                     for p in positions_pixels]
        instants = [self.convert_time_from_frames(t) for t in frames]
        # Cálculo de las velocidades.
        dx = np.diff(np.array(positions, dtype=np.float32), axis=0)
        dt = np.diff(np.array(instants, dtype=np.float32))
        v = [FloatVector2D(*dx[i] / dt[i]) for i in range(len(dx))]
        return v
