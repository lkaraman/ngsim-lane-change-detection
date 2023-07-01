import numpy as np
from shapely import LineString

from utils import VehicleFrame, SurroundingVehicles, Trajectory, VehicleFrameWithInterpolation, SemanticPosition, \
    SemanticFramesExtrapolation, SemanticFrames


class TrajectoryPredictor:

    def __init__(self, relevant_frames: dict[SemanticPosition, VehicleFrame], trajectory: Trajectory) -> None:

        self.frames = self.decorate_with_extrapolation(semantic_frames=relevant_frames, trajectory=trajectory)
        self.predicted_state = None

    def decorate_with_extrapolation(self, semantic_frames: SemanticFrames,
                                    trajectory: Trajectory) -> SemanticFramesExtrapolation:
        return {i: (VehicleFrameWithInterpolation(j) if i != SemanticPosition.EGO else VehicleFrameWithInterpolation(j,trajectory))
                for i, j in semantic_frames.items()

                }

    def _predict_for_dt(self, dt: float) -> SemanticFrames:
        return {
            i: j.extrapolate_with_constant_velocity(dt=dt) for i, j in self.frames.items()
        }

    def predict_for_dt(self, dt:float) -> None:
        self.predicted_state = self._predict_for_dt(dt=dt)



