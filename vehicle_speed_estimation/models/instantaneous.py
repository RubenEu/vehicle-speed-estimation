from enum import Enum
from typing import List, Tuple, Callable
import numpy as np

from simple_object_detection.typing import FloatVector2D
from simple_object_tracking.datastructures import TrackedObject, TrackedObjects

from vehicle_speed_estimation.estimation_model import EstimationModel, EstimationResults, \
    EstimationResult


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
        dx = np.diff(np.array(positions), axis=0)
        dt = np.diff(np.array(instants))
        v = [FloatVector2D(*dx[i] / dt[i]) for i in range(len(dx))]
        # Puesto que las velocidades son vectores, y la imagen tiene el eje Y al revés, para
        # estandarizarlo, cambiar el sentido del vector
        v = v * np.array([1., -1.])
        return v

    def fit(self, tracked_objects: TrackedObjects) -> EstimationResults:
        estimation_results = EstimationResults()
        # Realizar la estimación de cada objeto seguido.
        for tracked_object in tracked_objects:
            estimated_velocities = self.calculate_velocities(tracked_object)
            estimation = EstimationResult(estimated_velocities, tracked_object)
            # Añadir a la lista de estimaciones.
            estimation_results.add(estimation)
        return estimation_results


class InstantaneousVelocityWithKernelRegression(InstantaneousVelocity):
    """Estimación de las velocidades usando suavizado con regresión por kenels (Nadaraya-Watson).
    """

    class NadayaraWatsonEstimator(Enum):
        KERNEL_GAUSS = 0
        KERNEL_TRIANGULAR = 1
        KERNEL_QUADRATIC = 2

    def __init__(self,
                 kernel: NadayaraWatsonEstimator,
                 bandwidth: int = 1,
                 *args, **kwargs):
        """

        :param kernel: kernel utilizado para el suavizado.
        :param bandwidth: ancho usado en el kernel.
        :param purge_initial_estimations: cantidad de estimaciones iniciales que se eliminarán.
        :param purge_initial_estimations: cantidad de estimaciones finales que se eliminarán.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.kernel, self.kernel_derivated = self._get_kernel(kernel)
        self.bandwidth = bandwidth

    def calculate_velocities(self, tracked_object: TrackedObject) -> List[FloatVector2D]:
        """Realiza el cálculo de las velocidades y aplica un suavizado con regresión por kernels.

        :param tracked_object: seguimiento del objeto.
        :return: lista de las velocidades en cada instante.
        """
        # Variables de observación.
        ts = tracked_object.frames[1:]
        # Obtener las velocidades.
        vs = super().calculate_velocities(tracked_object)
        h = self.bandwidth
        # Aplicar Nadaraya-Watson para suavizarlas.
        velocities_smoothed = [self.nadayara_watson_estimator(ts[i], vs, ts, h, self.kernel)
                               for i in range(len(ts))]
        # Devolver velocidad suavizada en las unidades de la instancia de la clase.
        return velocities_smoothed

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
                NotImplemented,
        }
        return kernels[kernel]

    @staticmethod
    def kernel_gauss(t: float, t_i: float, h: float):
        """Kernel gaussiano.
        """
        return np.exp(- ((t - t_i) ** 2) / h)

    @staticmethod
    def kernel_gauss_derivated(t: float, t_i: float, h: float):
        """Derivada del kernel gaussiano.
        """
        return - (2 / h) * (t - t_i) * np.exp(- ((t - t_i) ** 2) / h)

    @staticmethod
    def kernel_triangular(t: float, t_i: float, h: float):
        """Kernel triangular.
        """
        assert abs(t - t_i) < h, f'|t-t_i| < h. {abs(t - t_i)} > {h}'
        return 1 - (abs(t - t_i) / h)

    @staticmethod
    def kernel_triangular_derivated(t: float, t_i: float, h: float):
        """Derivada del kernel triangular.
        """
        assert abs(t - t_i) < h, f'|t-t_i| < h. {abs(t - t_i)} > {h}'
        return - 1 / h * np.sign(t - t_i)

    @staticmethod
    def nadayara_watson_estimator(t: float,
                                  xs: List[FloatVector2D],
                                  ts: List[float],
                                  h: float,
                                  kernel: Callable[[float, float, float], float]) -> FloatVector2D:
        """Estimador de la posición aplicando Nadaraya-Watson.

        :param t: instante en el que se evalúa.
        :param xs: lista de posiciones.
        :param ts: lista de instantes.
        :param h: bandwidth.
        :param kernel: kernel utilizado.
        :return: lista de posiciones suavizadas.
        """
        xs = np.array(xs, dtype=np.float64)
        ts = np.array(ts, dtype=np.float64)
        indexes = list(range(len(ts)))
        _num = np.array([kernel(t, ts[i], h) * xs[i] for i in indexes]).sum(axis=0)
        _den = np.array([kernel(t, ts[i], h) for i in indexes]).sum(axis=0)
        result = _num / _den
        return FloatVector2D(*result)
