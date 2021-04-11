from abc import ABC, abstractmethod

from simple_object_tracking.typing import ObjectHistory


class EstimationModel(ABC):
    """

    TODO: Calcular lista de velocidades?
    """

    @abstractmethod
    def calculate_distance(self, object_history: ObjectHistory, **kwargs) -> float:
        """Calcula la distancia en píxeles.

        Este método debe ser implementado.

        :param object_history: historial del objeto [(frame, detección), ...]
        :param kwargs:
        :return: distancia calculada en píxeles.
        """

    @abstractmethod
    def calculate_time(self, object_history: ObjectHistory, **kwargs) -> float:
        """Calcula el tiempo transcurrido en frames, es decir, la cantidad de frames transcurridos.

        Este método debe ser implementado.

        :param object_history: historial del objeto [(frame, detección), ...]
        :param kwargs:
        :return: cantidad de frames transcurridos.
        """

    @abstractmethod
    def calculate_speed(self, object_history: ObjectHistory, **kwargs) -> float:
        """Calcula la velocidad en píxeles/frames.

        Este método debe ser implementado.

        :param object_history: historial del objeto [(frame, detección), ...]
        :param kwargs:
        :return: velocidad en píxeles/frames.
        """
