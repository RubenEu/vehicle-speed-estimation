from typing import List, Tuple, Callable, Any
from enum import Enum
import numpy as np

from simple_object_tracking.typing import ObjectHistory
from simple_object_tracking.utils import calculate_euclidean_distance

from vehicle_speed_estimation.estimation_model import EstimationModel
from vehicle_speed_estimation.utils.conversion import SECOND_TO_HOUR


class InstantaneousSpeed(EstimationModel):
    """

    """
    def calculate_speeds(self, object_history: ObjectHistory):
        speeds = self._calculate_speeds_px_frame(object_history)
        if self.units == self.Units.PIXELS_FRAME:
            ...  # no hacer nada
        elif self.units == self.Units.METERS_SECOND:
            ...  # TODO
        elif self.units == self.Units.KILOMETERS_HOURS:
            speeds = [self.speed_vector_px_frame_to_kmh(speed) for speed in speeds]
        return speeds

    def _calculate_speeds_px_frame(self, object_history: ObjectHistory) -> List[Tuple[float, float]]:
        """

        :param object_history:
        :return:
        """
        # Desempaquetar información.
        frames, detections = zip(*object_history)
        positions = [self.get_object_point(detection) for detection in detections]
        # Usando:
        # https://stackoverflow.com/questions/16841729/how-do-i-compute-the-derivative-of-an-array-in-python/19459160
        dx = np.diff(np.array(positions, dtype=np.float32), axis=0)
        dt = np.diff(np.array(frames, dtype=np.float32))
        v = [tuple(dx[i] / dt[i]) for i in range(len(dx))]
        return v
        # # Calcular los vectores de velocidad.
        # index_prev = 0
        # speeds = list()
        # for index in range(1, len(positions)):
        #     # Cálculo de distancia.
        #     position_prev = np.array(positions[index_prev])
        #     position = np.array(positions[index])
        #     distance = position - position_prev
        #     # Cálculo de tiempo.
        #     time = frames[index] - frames[index_prev]
        #     # Cálculo de velocidad.
        #     speed = distance / time
        #     # Añadir a la lista.
        #     speeds.append(tuple(speed))
        #     # Actualizar la posición previa.
        #     index_prev = index
        # return speeds

    def calculate_positions_px(self, object_history: ObjectHistory) -> List[Tuple[float, float]]:
        _, detections = zip(*object_history)
        positions = [self.get_object_point(detection) for detection in detections]
        return positions

    def calculate_positions_meter(self, object_history: ObjectHistory) -> List[Tuple[float, float]]:
        positions_px = self.calculate_positions_px(object_history)
        positions_meter = [(position_px[0] * self.pixel_to_meters[0],
                            position_px[1] * self.pixel_to_meters[1])
                           for position_px in positions_px]
        return positions_meter

    def calculate_instants_second(self, object_history: ObjectHistory) -> List[float]:
        """

        :param object_history:
        :return:
        """
        frames, _ = zip(*object_history)
        # Convertir el frame a segundos.
        instants_s = [frame * self.seconds_per_frame for frame in frames]
        return instants_s

    def calculate_instants_hour(self, object_history: ObjectHistory) -> List[float]:
        """

        :param object_history:
        :return:
        """
        instants_s = self.calculate_instants_second(object_history)
        instants_h = [instant_s * SECOND_TO_HOUR for instant_s in instants_s]
        return instants_h


class InstantaneosSpeedWithKernelRegression(InstantaneousSpeed):

    class NadayaraWatsonEstimator(Enum):
        KERNEL_GAUSS = 0
        KERNEL_TRIANGULAR = 1
        KERNEL_QUADRATIC = 2

    def __init__(self, kernel: NadayaraWatsonEstimator, bandwidth: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bandwidth = bandwidth
        self.kernel, self.kernel_derivated = self.get_kernel(kernel)

    def _calculate_speeds_px_frame(self, object_history: ObjectHistory) -> List[Tuple[float, float]]:
        # Desempaquetar información.
        frames, detections = zip(*object_history)
        positions = [self.get_object_point(detection) for detection in detections]
        # Calcular los vectores de velocidad.
        index_prev = 0
        speeds = list()
        for index in range(0, len(positions)):
            # Posición previa ya ctual..
            position_prev = np.array(positions[index_prev])
            position = np.array(positions[index])
            # Variables necesarias
            ts = frames
            xs = np.array([np.array(position) for position in positions])
            h = self.bandwidth
            N = range(0, len(positions))
            # Cálculo de velocidad.
            _num_1 = np.array([self.kernel_gauss_derivated(ts[index], ts[i], h) * xs[i] for i in N])
            _num_2 = np.array([self.kernel_gauss(ts[index], ts[i], h) for i in N])
            _num_3 = np.array([self.kernel_gauss_derivated(ts[index], ts[i], h) for i in N])
            _num_4 = np.array([self.kernel_gauss(ts[index], ts[i], h) * xs[i] for i in N])
            _num = _num_1.sum(axis=0) * _num_2.sum(axis=0) - _num_3.sum(axis=0) * _num_4.sum(axis=0)
            _den = np.array([self.kernel_gauss(ts[index], ts[i], h) for i in N]).sum(axis=0) ** 2

            v = _num / _den
            # Añadir a la lista.
            speeds.append(tuple(v))
            # Actualizar la posición previa.
            index_prev = index
        return speeds

    def get_kernel(self, kernel: NadayaraWatsonEstimator) -> Tuple[Callable, Callable]:
        kernels = {
            self.NadayaraWatsonEstimator.KERNEL_GAUSS:
                (self.kernel_gauss, self.kernel_gauss_derivated),
        }
        return kernels[kernel]

    @staticmethod
    def kernel_gauss(t, t_i, h):
        return np.exp(-((t - t_i) ** 2 / h))

    @staticmethod
    def kernel_gauss_derivated(t, t_i, h):
        return -(2 / h) * (t - t_i) * np.exp(-((t - t_i) ** 2) / h)

    @staticmethod
    def nadayara_watson_estimator(index, positions, instants, kernel, bandwidth):
        x = np.array([np.array(position) for position in positions])
        t = np.array(instants)
        indexes = list(range(len(positions)))
        _num = np.array([kernel(t[index], t[i], bandwidth) * x[i] for i in indexes]).sum(axis=0)
        _den = np.array([kernel(t[index], t[i], bandwidth) for i in indexes]).sum(axis=0)
        return _num / _den
