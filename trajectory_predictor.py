import itertools

from utils import VehicleFrame, Trajectory, VehicleFrameWithInterpolation, SemanticPosition, \
    SemanticFramesExtrapolation, SemanticFrames
from utils_collision import rotated_rectangles_intersect


class TrajectoryPredictor:

    def __init__(self, relevant_frames: SemanticFrames, trajectory: Trajectory) -> None:

        self.frames = self.decorate_with_extrapolation(semantic_frames=relevant_frames, trajectory=trajectory)
        self.predicted_states: list[SemanticFrames] = [relevant_frames]
        self.collision_info: list = []

    def decorate_with_extrapolation(self, semantic_frames: SemanticFrames,
                                    trajectory: Trajectory) -> SemanticFramesExtrapolation:
        return {i: (VehicleFrameWithInterpolation(j) if i != SemanticPosition.EGO else VehicleFrameWithInterpolation(j,
                                                                                                                     trajectory))
                for i, j in semantic_frames.items()

                }

    def _predict_for_dt(self, dt: float) -> SemanticFrames:
        return {
            i: j.extrapolate_with_constant_velocity(dt=dt) for i, j in self.frames.items()
        }

    def predict_for_dt(self, dt: float) -> None:
        self.predicted_states.append(self._predict_for_dt(dt=dt))

    def is_collision_in_predicted(self):

        collision_info = []

        for v1, v2 in itertools.combinations(self.predicted_states[-1].items(), 2):
            i1, f1 = v1
            i2, f2 = v2

            pos1 = ([f1.s, f1.d], f1.length, f1.width, f1.yaw)
            pos2 = ([f2.s, f2.d], f2.length, f2.width, f2.yaw)

            if rotated_rectangles_intersect(rect1=pos1, rect2=pos2) and (f1.is_ego or f2.is_ego):
                collision_info.append((i1, i2))

        self.collision_info.extend(collision_info)


