from abc import ABC, abstractmethod
from enum import Enum
from typing import List
import numpy as np

from simple_object_detection.object import Object
from simple_object_detection.typing import Point2D, FloatVector2D
from simple_object_tracking.datastructures import TrackedObjects, TrackedObject


class EstimationResult:
    """Resultados de una estimación.
    """
    def __init__(self, velocities: List[FloatVector2D], tracked_object: TrackedObject):
        """

        :param velocities: lista de velocidades.
        :param tracked_object: objeto seguido.
        """
        self.velocities = velocities
        self.tracked_object = tracked_object

    def __str__(self) -> str:
        return f'EstimationResult(id={self.tracked_object.id}, ' \
               f'mean_speed={self.mean_speed()}, ' \
               f'mean_velocity={self.mean_velocity()})'

    def __repr__(self) -> str:
        return str(self)

    def mean_velocity(self) -> FloatVector2D:
        """Calcula el vector de velocidad media.

        :return: vector de velocidad media.
        """
        return FloatVector2D(*np.array(self.velocities).mean(axis=0))

    def mean_speed(self) -> float:
        """Calcula el módulo de la velocidad media.

        :return: módulo de la velocidad media.
        """
        mean_velocity = np.array(self.velocities).mean(axis=0)
        return np.linalg.norm(mean_velocity)


class EstimationResults:
    """Resultados de un modelo de estimación.

    TODO: Tener en cuenta vehículos válidos.
    """
    def __init__(self):
        self._estimations_results: List[EstimationResult] = []
        self._ignored_ids: List[int] = []

    def __getitem__(self, item: int) -> EstimationResult:
        """Devuelve la estimación item-ésima.

        :param item: índice de la estimación.
        :return: resultado de la estimación.
        """
        if item >= len(self._estimations_results):
            raise IndexError(f'El índice {item} está fuera del límite.')
        return self._estimations_results[item]

    def __len__(self) -> int:
        """Cantidad de estimaciones resultantes.
        """
        return len(self._estimations_results)

    def __str__(self) -> str:
        return f'EstimationResults({len(self._estimations_results)})'

    def __repr__(self) -> str:
        return str(self)

    def add(self, estimation_result: EstimationResult) -> None:
        """Añade un resultado de una estimación.

        :param estimation_result: estimación.
        :return: None.
        """
        self._estimations_results.append(estimation_result)

    def ignore_object(self, id_: int) -> None:
        """Se ignorará el objeto indicado en los métodos de cálculo sobre las estimaciones, como por
        ejemplo el cálculo del MSE.

        :param id_: índice del objeto.
        :return: None.
        """
        self._ignored_ids.append(id_)

    def ignore_objects(self, ids: List[int]) -> None:
        """Añade en objetos ignorados la lista de ids.

        :param ids: ids de los objetos para ignorar.
        :return: None.
        """
        for id_ in ids:
            self.ignore_object(id_)

    def object_ignored(self, id_: int) -> bool:
        """Comprueba si un objeto está siendo ignorado.

        :param id_: índice del objeto.
        :return: si está siendo ignorado o no.
        """
        return id_ in self._ignored_ids

    def velocity_mse(self, expected_) -> FloatVector2D:
        raise NotImplemented()

    def speed_mse(self, expected_speeds: List[float]) -> float:
        """Calcula el error cuadrático medio del módulo de las velocidades.

        :param expected_speeds: lista con las velocidades esperadas.
        :return: error cuadrático medio.
        """
        # Comprobar que tienen las mismas dimensiones.
        if len(self._estimations_results) != len(expected_speeds):
            raise Exception('La lista de velocidades introducida no tiene la dimensión esperada.')
        # Filtrar los índices ignorados.
        estimated_speeds = [result.mean_speed() for result in self._estimations_results
                            if not self.object_ignored(result.tracked_object.id)]
        expected_speeds = [speed for id_, speed in enumerate(expected_speeds)
                           if not self.object_ignored(id_)]
        # Calcular MSE.
        estimated_speeds = np.array(estimated_speeds)
        expected_speeds = np.array(expected_speeds)
        return ((estimated_speeds - expected_speeds) ** 2).mean(axis=0)


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
        elif self.time_unit == self.TimeUnit.HOUR:
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

    def convert_velocity_from_pixels_frames(self, velocity: FloatVector2D) -> FloatVector2D:
        """Convierte el vector de velocidad en píxeles / frames a la unidad especificada al
        instanciar la clase.

        :param velocity: vector de velocidad en píxeles/frames.
        :return: vector de velocidad en la unidad especificada al instanciar la clase.
        """
        # Convertir unidad de espacio.
        velocity_x, velocity_y = velocity
        if self.length_unit == self.LengthUnit.METER:
            velocity_x = velocity_x * self.pixel_to_meters.x
            velocity_y = velocity_y * self.pixel_to_meters.y
        elif self.length_unit == self.LengthUnit.KILOMETER:
            velocity_x = velocity_x * self.pixel_to_meters.x / 1000
            velocity_y = velocity_y * self.pixel_to_meters.y / 1000
        # Convertir la unidad de tiempo.
        if self.time_unit == self.TimeUnit.SECOND:
            velocity_x = velocity_x * self.frames_per_second
            velocity_y = velocity_y * self.frames_per_second
        elif self.time_unit == self.TimeUnit.HOUR:
            velocity_x = velocity_x * self.frames_per_second * 3600
            velocity_y = velocity_y * self.frames_per_second * 3600
        # Devolver vector de velocidad.
        return FloatVector2D(velocity_x, velocity_y)

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

    @abstractmethod
    def fit(self, tracked_objects: TrackedObjects) -> EstimationResults:
        """Realizar la estimación del modelo.

        :param tracked_objects: estructura con los seguimientos de los objetos.
        :return: estimaciones calculadas.
        """
