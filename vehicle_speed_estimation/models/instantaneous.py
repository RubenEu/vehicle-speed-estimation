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
        dx = np.diff(np.array(positions, dtype=np.float32), axis=0)
        dt = np.diff(np.array(instants, dtype=np.float32))
        v = [FloatVector2D(*dx[i] / dt[i]) for i in range(len(dx))]
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

    Problema: si se realizan los cálculos con el tiempo en horas, las unidades serán del orden de
    10e-3 e incluso 10e-5, por tanto, el bandwidth necesitará ser proporcional a él. Un kernel mayor
    que de 10e-5 hará que el suavizado sea excesivo. Por tanto, el cálculo de las velocidades se
    realiza con píxeles y frames como unidades, y la conversión se realiza sobre la estimación de
    velocidad y no sobre las unidades de entrada.

    Si se utiliza un kernel pequeño cuando la diferencia entre los instantes de tiempo es grande,
    habrá cálculos que puedan resultar en underflow.
    """

    class NadayaraWatsonEstimator(Enum):
        KERNEL_GAUSS = 0
        KERNEL_GAUSS_2 = 1
        KERNEL_TRIANGULAR = 2
        KERNEL_QUADRATIC = 3

    def __init__(self,
                 kernel: NadayaraWatsonEstimator,
                 bandwidth: int = 1,
                 purge_initial_estimations: int = 0,
                 purge_final_estimations: int = 0,
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
        self.purge_initial_estimations = purge_initial_estimations
        self.purge_final_estimations = purge_final_estimations

    def calculate_velocities(self, tracked_object: TrackedObject) -> List[FloatVector2D]:
        """Realiza el cálculo de las velocidades en cada instante que fue detectado el objeto.

        La unidad de medida de las velocidades vendrá determinada por la introducida al instanciar
        el modelo con el parámetro `units`.

        :param tracked_object: seguimiento del objeto.
        :return: lista de las velocidades en cada instante.
        """
        # Variables observadas.
        frames = tracked_object.frames
        positions = [self.get_object_point(detection) for detection in tracked_object.detections]
        # Variables para el cálculo con el estimador de Nadaraya-Watson.
        ts = frames
        xs = np.array(positions)
        # Variables del modelo.
        kernel = self.kernel
        kernel_derivated = self.kernel_derivated
        h = self.bandwidth
        # Calcular los vectores de velocidad.
        velocities = list()
        for t in ts:
            # i-índices.
            indexes = range(0, len(ts))
            # Calcular el numerador y denominador por partes.
            n1 = np.array([kernel_derivated(t, ts[i], h) * xs[i] for i in indexes]).sum(axis=0)
            n2 = np.array([kernel(t, ts[i], h) for i in indexes]).sum(axis=0)
            n3 = np.array([kernel_derivated(t, ts[i], h) for i in indexes]).sum(axis=0)
            n4 = np.array([kernel(t, ts[i], h) * xs[i] for i in indexes]).sum(axis=0)
            n = (n1 * n2) - (n3 * n4)
            d = np.array([kernel(t, ts[i], h) for i in indexes]).sum(axis=0) ** 2
            # Cálculo del vector de velocidad.
            v = n / d
            # Convertir las unidades a las indicadas al instanciar la c.ase
            v = self.convert_velocity_from_pixels_frames(FloatVector2D(*v))
            # Añadir a la lista.
            velocities.append(v)
        return self._purge_extremes_estimations(velocities)

    def _purge_extremes_estimations(self, velocities: List[FloatVector2D]) -> List[FloatVector2D]:
        """Elimina las estimaciones iniciales y finales para evitar que el error que produce el
        suavizado en los extremos repercuta en el cálculo de la velocidad media.

        :param velocities: vector de velocidades.
        :return: vector de velocidades purgado.
        """
        start = self.purge_initial_estimations
        end = len(velocities) - self.purge_final_estimations
        return velocities[start:end]

    def _get_kernel(self, kernel: NadayaraWatsonEstimator) -> Tuple[Callable, Callable]:
        """Devuelve las funciones del kernel y su derivada respectivamente.

        :param kernel: kernel.
        :return: función del kernel y función del kernel derivado.
        """
        kernels = {
            self.NadayaraWatsonEstimator.KERNEL_GAUSS:
                (self.kernel_gauss, self.kernel_gauss_derivated),
            self.NadayaraWatsonEstimator.KERNEL_GAUSS_2:
                (self.kernel_gauss_2, self.kernel_gauss_derivated_2),
            self.NadayaraWatsonEstimator.KERNEL_TRIANGULAR:
                (self.kernel_triangular, self.kernel_triangular_derivated),
            self.NadayaraWatsonEstimator.KERNEL_QUADRATIC:
                None
        }
        return kernels[kernel]

    @staticmethod
    def kernel_gauss(t, t_i, h):
        return np.exp(- ((t - t_i) ** 2) / h)

    @staticmethod
    def kernel_gauss_derivated(t, t_i, h):
        return - (2 / h) * (t - t_i) * np.exp(- ((t - t_i) ** 2) / h)

    @staticmethod
    def kernel_gauss_2(t, t_i, h):
        return np.exp(- ((t - t_i) ** 2) / (2 * h ** 2))

    @staticmethod
    def kernel_gauss_derivated_2(t, t_i, h):
        return - ((2 * t - 2 * t_i) / (2 * h ** 2)) * np.exp(- ((t - t_i) ** 2) / (2 * h ** 2))

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
