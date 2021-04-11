from simple_object_tracking.typing import ObjectHistory
from vehicle_speed_estimation.estimation_model import EstimationModel


class DistanceAverage(EstimationModel):
    """



    """

    def calculate_distance(self, object_history: ObjectHistory, **kwargs) -> float:
        pass

    def calculate_time(self, object_history: ObjectHistory, **kwargs) -> float:
        pass

    def calculate_speed(self, object_history: ObjectHistory, **kwargs) -> float:
        pass
