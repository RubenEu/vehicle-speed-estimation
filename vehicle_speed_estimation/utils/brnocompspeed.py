from copy import copy

from tqdm import tqdm
from typing import Tuple, List, Dict, NamedTuple

from simple_object_detection.typing import Point2D
from simple_object_tracking.datastructures import TrackedObjects, TrackedObject


def map_dataset_to_observations(dataset: dict,
                                tracked_objects: TrackedObjects,
                                tracked_objects_h: TrackedObjects,
                                line: Tuple[Point2D, Point2D],
                                max_time_diff: float = 0.4) -> Tuple[List[Dict],
                                                                     TrackedObjects,
                                                                     TrackedObjects]:
    """Función para establecer una correspondencia entre los vehículos detectados en el seguimiento
    y los vehículos anotados en el dataset.

    :param dataset: información de la sesión del dataset BrnoCompSpeed.
    :param tracked_objects: estructura de datos de los objetos seguidos.
    :param tracked_objects_h: estructura de datos de los objetos seguidos con la homografía aplicada.
    :param line: línea que se usará para comparar el instante en que pasaron los vehículos (anotados
    y seguidos).
    :param max_time_diff: tiempo máximo de diferencia que puede haber entre un vehículo anotado y un
    vehículo detectado al pasar la línea indicada.
    :return: lista de los vehículos anotados y estructura de los objetos seguidos indexadas en el
    mismo orden de correspondencia.
    """
    fps: float = dataset['fps']
    cars: List[Dict] = dataset['cars'].copy()
    tracked_objects_h_list: List[TrackedObject] = list(tracked_objects_h)
    # 1. Reordenar tracked_objects_h por su orden de llegada a la línea final.
    tracked_objects_h_list.sort(key=lambda tobj: tobj.find_closest_detection_to_line(line).frame)
    # 2. Crear el diccionario de coches anotados y la lista de objetos seguidos con el
    # emparejamiento correcto.
    dataset_cars_ids_matched: List[int] = []
    tracked_objects_matched: List[TrackedObject] = []
    tracked_objects_h_matched: List[TrackedObject] = []
    cars_matched: List[Dict] = []
    id_ = 0
    t = tqdm(total=len(tracked_objects_h_list), desc='Mapping dataset to tracked objects.')
    for tracked_object_h in tracked_objects_h_list:
        time_passed_line = tracked_object_h.find_closest_detection_to_line(line).frame / fps
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
            tracked_object = copy(tracked_objects[tracked_object_h.id])
            tracked_object_h = copy(tracked_object_h)
            # Asignar el nuevo id y actualizarlo.
            candidate['carId'] = id_
            tracked_object.id = id_
            tracked_object_h.id = id_
            id_ += 1
            # Realizar el emparejamiento.
            tracked_objects_matched.append(tracked_object)
            tracked_objects_h_matched.append(tracked_object_h)
            cars_matched.append(candidate)
        t.update()
    tracked_objects_mapped = TrackedObjects()
    tracked_objects_mapped.tracked_objects = tracked_objects_matched
    tracked_objects_h_mapped = TrackedObjects()
    tracked_objects_h_mapped.tracked_objects = tracked_objects_h_matched
    return cars_matched, tracked_objects_mapped, tracked_objects_h_mapped
