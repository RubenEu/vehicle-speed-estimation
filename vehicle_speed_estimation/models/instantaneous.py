from typing import List, Tuple, Callable, Any
from enum import Enum
import numpy as np

from simple_object_tracking.typing import ObjectHistory

from vehicle_speed_estimation.estimation_model import EstimationModel


class InstantaneousVelocity(EstimationModel):
    """Estimación de los vectores de velocidad usando diferenciación discreta.
    """

    def calculate_velocities(self, object_history: ObjectHistory) -> List[Tuple[float, float]]:
        """Realiza el cálculo de las velocidades en cada instante que fue detectado el objeto.

        La unidad de medida de las velocidades vendrá determinada por la introducida al instanciar
        el modelo con el parámetro `units`.

        :param object_history: historial del objeto.
        :return: lista de las velocidades en cada instante.
        """
        speeds = self._calculate_velocities(object_history)
        if self.units == self.Units.PIXELS_FRAME:
            ...  # no hacer nada
        elif self.units == self.Units.METERS_SECOND:
            ...  # TODO
        elif self.units == self.Units.KILOMETERS_HOURS:
            speeds = [self.speed_vector_px_frame_to_kmh(speed) for speed in speeds]
        return speeds

    def _calculate_velocities(self, object_history: ObjectHistory) -> List[Tuple[float, float]]:
        """Realiza el cálculo de las velocidades en cada instante en la imagen. La unidad de medida
        de la distancia es píxeles y la del tiempo, frames.

        :param object_history: historial del objeto.
        :return: lista de las velocidades en cada instante en píxels/frame.
        """
        # Desempaquetar información.
        frames, detections = zip(*object_history)
        positions = [self.get_object_point(detection) for detection in detections]
        # Cálculo de las velocidades.
        dx = np.diff(np.array(positions, dtype=np.float32), axis=0)
        dt = np.diff(np.array(frames, dtype=np.float32))
        v = [tuple(dx[i] / dt[i]) for i in range(len(dx))]
        return v


class InstantaneousVelocityWithKernelRegression(InstantaneousVelocity):
    """Estimación de las velocidades usando suavizado con regresión por kenels (Nadaraya-Watson).
    """

    class NadayaraWatsonEstimator(Enum):
        KERNEL_GAUSS = 0
        KERNEL_TRIANGULAR = 1
        KERNEL_QUADRATIC = 2

    def __init__(self, kernel: NadayaraWatsonEstimator, bandwidth: int = 1, *args, **kwargs):
        """

        :param kernel: kernel utilizado para el suavizado.
        :param bandwidth: ancho usado en el kernel.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.bandwidth = bandwidth
        self.kernel, self.kernel_derivated = self._get_kernel(kernel)

    def _calculate_velocities(self, object_history: ObjectHistory) -> List[Tuple[float, float]]:
        """Realiza el cálculo de las velocidades en cada instante en la imagen. La unidad de medida
        de la distancia es píxeles y la del tiempo, frames.

        :param object_history: historial del objeto.
        :return: lista de las velocidades en cada instante en píxels/frame.
        """
        # Desempaquetar información.
        frames, detections = zip(*object_history)
        # Variables observadas.
        ts = frames
        xs = np.array([np.array(self.get_object_point(detection)) for detection in detections])
        # Variables del modelo.
        kernel = self.kernel
        kernel_ = self.kernel_derivated
        h = self.bandwidth
        # Calcular los vectores de velocidad.
        speeds = list()
        for t in frames:
            # i-índices.
            indexes = range(0, len(frames))
            # Calcular el numerador y denominador por partes.
            n1 = np.array([kernel_(t, ts[i], h) * xs[i] for i in indexes]).sum(axis=0)
            n2 = np.array([kernel(t, ts[i], h) for i in indexes]).sum(axis=0)
            n3 = np.array([kernel_(t, ts[i], h) for i in indexes]).sum(axis=0)
            n4 = np.array([kernel(t, ts[i], h) * xs[i] for i in indexes]).sum(axis=0)
            n = (n1 * n2) - (n3 * n4)
            d = np.array([kernel(t, ts[i], h) for i in indexes]).sum(axis=0) ** 2
            # Cálculo del vector de velocidad.
            v = n / d
            # Añadir a la lista.
            speeds.append(tuple(v))
        return speeds

    def _get_kernel(self, kernel: NadayaraWatsonEstimator) -> Tuple[Callable, Callable]:
        """Devuelve las funciones del kernel y su derivada respectivamente.

        :param kernel: kernel.
        :return: función del kernel y función del kernel derivado.
        """
        kernels = {
            self.NadayaraWatsonEstimator.KERNEL_GAUSS:
                (self.kernel_gauss, self.kernel_gauss_derivated),
            self.NadayaraWatsonEstimator.KERNEL_TRIANGULAR:
                (self.kernel_triangular, self.kernel_triangular_derivated),
            self.NadayaraWatsonEstimator.KERNEL_QUADRATIC:
                None
        }
        return kernels[kernel]

    @staticmethod
    def kernel_gauss(t, t_i, h):
        return np.exp(- (t - t_i) ** 2 / h)

    @staticmethod
    def kernel_gauss_derivated(t, t_i, h):
        return - 2 / h * (t - t_i) * np.exp(- (t - t_i) ** 2 / h)

    @staticmethod
    def kernel_triangular(t, t_i, h):
        assert abs(t - t_i) < h, f'|t-t_i| < h. {abs(t - t_i)} > {h}'
        return 1 - (abs(t - t_i) / h)

    @staticmethod
    def kernel_triangular_derivated(t, t_i, h):
        assert abs(t - t_i) < h, f'|t-t_i| < h. {abs(t - t_i)} > {h}'
        return - 1 / h * np.sign(t - t_i)

    @staticmethod
    def nadayara_watson_estimator(t, xs, ts, h, kernel):
        """Estimador de la posición aplicando Nadaraya-Watson.

        :param t: instante en el que se evalúa.
        :param xs: lista de posiciones.
        :param ts: lista de instantes.
        :param h: bandwidth.
        :param kernel: kernel utilizado.
        :return: lista de posiciones suavizadas.
        """
        xs = np.array([np.array(x) for x in xs])
        ts = np.array(ts)
        indexes = list(range(len(ts)))
        _num = np.array([kernel(t, ts[i], h) * xs[i] for i in indexes]).sum(axis=0)
        _den = np.array([kernel(t, ts[i], h) for i in indexes]).sum(axis=0)
        return _num / _den
