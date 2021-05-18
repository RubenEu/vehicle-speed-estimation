from copy import copy

from tqdm import tqdm
from typing import Tuple, List, Dict, NamedTuple

from simple_object_detection.typing import Point2D
from simple_object_tracking.datastructures import TrackedObjects, TrackedObject


def map_dataset_to_observations(dataset: dict,
                                tracked_objects: TrackedObjects,
                                line: Tuple[Point2D, Point2D],
                                max_time_diff: float = 0.4) -> Tuple[List[Dict], TrackedObjects]:
    """Función para establecer una correspondencia entre los vehículos detectados en el seguimiento
    y los vehículos anotados en el dataset.

    TODO:
        - Añadir estructuras de ``TrackedObjects`` extra para realizar la ordenación en ellas
        también. Esto es útil para reordenar tanto las que poseen la homografía, como las que no,
        ya que la ordenación se realiza sobre la homografiada.

    :param dataset: información de la sesión del dataset BrnoCompSpeed.
    :param tracked_objects: estructura de datos de los objetos seguidos.
    :param line: línea que se usará para comparar el instante en que pasaron los vehículos (anotados
    y seguidos).
    :param max_time_diff: tiempo máximo de diferencia que puede haber entre un vehículo anotado y un
    vehículo detectado al pasar la línea indicada.
    :return: lista de los vehículos anotados y estructura de los objetos seguidos indexadas en el
    mismo orden de correspondencia.
    """
    fps: float = dataset['fps']
    cars: List[Dict] = dataset['cars'].copy()
    tracked_objects_list: List[TrackedObject] = list(tracked_objects)
    # 1. Reordenar tracked_objects por su orden de llegada a la línea final.
    tracked_objects_list.sort(key=lambda tobj: tobj.find_closest_detection_to_line(line).frame)
    # 2. Crear el diccionario de coches anotados y la lista de objetos seguidos con el
    # emparejamiento correcto.
    dataset_cars_ids_matched: List[int] = []
    tracked_objects_matched: List[TrackedObject] = []
    cars_matched: List[Dict] = []
    id_ = 0
    t = tqdm(total=len(tracked_objects_list), desc='Mapping dataset to tracked objects.')
    for tracked_object in tracked_objects_list:
        time_passed_line = tracked_object.find_closest_detection_to_line(line).frame / 50.0
        # Vehículos del dataset candidatos.
        candidates = [car for car in cars
                      if car['carId'] not in dataset_cars_ids_matched and
                      abs(car['timeIntersectionLastShifted'] - time_passed_line) < max_time_diff]
        # Ordenarlos por el que pasó en el instante más cercano y realizar el emparejamiento con él.
        if len(candidates) > 0:
            candidates.sort(
                key=lambda car: abs(car['timeIntersectionLastShifted'] - time_passed_line))
            candidate = candidates[0]
            # Marcar el candidato como emparejado.
            dataset_cars_ids_matched.append(candidate['carId'])
            # Copiar los datos para editar sus ids y mantener los originales.
            candidate = candidate.copy()
            tracked_object = copy(tracked_object)
            # Asignar el nuevo id y actualizarlo.
            candidate['carId'] = id_
            tracked_object.id = id_
            id_ += 1
            # Realizar el emparejamiento.
            tracked_objects_matched.append(tracked_object)
            cars_matched.append(candidate)
        t.update()
    tracked_objects_mapped = TrackedObjects()
    tracked_objects_mapped.tracked_objects = tracked_objects_matched
    return cars_matched, tracked_objects_mapped
