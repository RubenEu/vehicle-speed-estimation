from typing import Tuple
import numpy as np

from simple_object_tracking.typing import ObjectHistory
from simple_object_tracking.utils import calculate_euclidean_distance

from vehicle_speed_estimation.estimation_model import EstimationModel


class TwoPositionsSpeedAverage(EstimationModel):

    def calculate_distance_px(self, object_history: ObjectHistory, initial_index: int = 0,
                              final_index: int = -1) -> Tuple[float, float]:
        _, initial_detection = object_history[initial_index]
        _, final_detection = object_history[final_index]
        initial_pos = self.get_object_point(initial_detection)
        final_pos = self.get_object_point(final_detection)
        distance_vector = tuple(abs(np.array(final_pos) - np.array(initial_pos)))
        return distance_vector

    def calculate_time_frames(self, object_history: ObjectHistory, initial_index: int = 0,
                              final_index: int = -1) -> float:
        initial_frame, _ = object_history[initial_index]
        final_frame, _ = object_history[final_index]
        return final_frame - initial_frame

    def calculate_speed_kmh(self, object_history: ObjectHistory, initial_index: int = 0,
                            final_index: int = -1) -> Tuple[float, float]:
        distance_px = self.calculate_distance_px(object_history, initial_index, final_index)
        time_frames = self.calculate_time_frames(object_history, initial_index, final_index)
        speed_px_frame = distance_px[0] / time_frames, distance_px[1] / time_frames
        speed_km_h = self.speed_vector_px_frame_to_kmh(speed_px_frame)
        return speed_km_h
