from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Optional
from shapely import LineString

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Lane:
    x: List[float]
    y: List[float]


@dataclass
class ReferenceLine:
    x: List[float]
    y: List[float]


@dataclass
class Road:
    reference_line: ReferenceLine
    lanes: Dict[str, Lane]


@dataclass
class Vehicle:
    object_id: int
    s: List[float]
    d: List[float]
    times: List[float]
    frames: List[int]
    length: List[float]
    width: List[float]
    vs: List[float]
    vd: List[float]
    lanes: List[int]

    def plot(self, var_name):
        v = getattr(self, var_name)
        plt.plot(self.times, v)
        plt.show()


@dataclass
class Traffic:
    vehicles: List[Vehicle]

    def find_vehicle_with_id_and_start_and_end_frames(self, veh_id: str, frame_s: int, frame_end: int):
        l = [v for v in self.vehicles if v.object_id == veh_id and v.frames[0] == frame_s and v.frames[-1] == frame_end]
        assert len(l) == 1
        return l[0]

    def find_vehicle_with_id_and_containing_frames(self, veh_id: int, frame_s: int, frame_end: int) -> Vehicle:
        l = [v for v in self.vehicles if v.object_id == veh_id and v.frames[0] <= frame_s and v.frames[-1] >= frame_end]
        assert len(l) == 1
        return l[0]


@dataclass
class Scenario:
    road: Road
    traffic: Traffic


@dataclass
class AnnotationEntry:
    vehicle_id: int
    index_start: int
    index_end: int


class SemanticPosition(Enum):
    EGO = auto()
    SAME_FRONT = auto()
    SAME_BACK = auto()
    NEXT_FRONT = auto()
    NEXT_BACK = auto()
    OTHER = auto()


@dataclass
class VehicleFrame:
    object_id: int
    s: float
    d: float
    lane: int
    is_ego: bool
    width: float
    length: float
    velocity: float
    yaw: Optional[float] = 0


vehicle_far_far_away = VehicleFrame(
    object_id=-1,
    s=10000000,
    d=10000000,
    lane=0,
    is_ego=False,
    width=0,
    length=0,
    velocity=0,
    yaw=0
)


@dataclass
class Trajectory:
    x: list[float]
    y: list[float]

    def __post_init__(self):
        self.shp_traj = LineString([(i, j) for i, j in zip(self.x, self.y)])

    def extrapolate(self, dt: float, v: float) -> tuple[float, float, float]:
        ds = v * dt

        r = self.shp_traj.interpolate(distance=ds)
        r_lookahead = self.shp_traj.interpolate(distance=ds + 0.5)

        x, y = r.x, r.y
        ang = np.arctan2(r_lookahead.y - r.y, r_lookahead.x - r.x)

        return x, y, ang


@dataclass
class VehicleFrameWithInterpolation:
    vehicle_frame: VehicleFrame
    trajectory: Optional[Trajectory] = None

    def extrapolate_with_constant_velocity(self, dt: float) -> VehicleFrame:
        v = self.vehicle_frame.velocity

        if self.trajectory is None:
            s = self.vehicle_frame.s + self.vehicle_frame.velocity * dt
            d = self.vehicle_frame.d
            yaw = 0

        else:
            s, d, yaw = self.trajectory.extrapolate(dt=dt, v=v)

        return VehicleFrame(
            object_id=self.vehicle_frame.object_id,
            s=s,
            d=d,
            lane=self.vehicle_frame.lane,
            is_ego=self.vehicle_frame.is_ego,
            width=self.vehicle_frame.width,
            length=self.vehicle_frame.length,
            velocity=self.vehicle_frame.velocity,
            yaw=yaw
        )


SemanticFrames = dict[SemanticPosition, VehicleFrame]
SemanticFramesExtrapolation = dict[SemanticPosition, VehicleFrameWithInterpolation]


@dataclass(frozen=True)
class SurroundingVehicleFrames:
    vehicle_current_lane_front: VehicleFrame
    vehicle_current_lane_back: VehicleFrame
    vehicle_next_lane_front: VehicleFrame
    vehicle_next_lane_back: VehicleFrame
    ego: VehicleFrame

    def to_gen(self):
        yield self.vehicle_current_lane_front
        yield self.vehicle_current_lane_back
        yield self.vehicle_next_lane_front
        yield self.vehicle_next_lane_back

    def as_dict(self) -> SemanticFrames:
        return {
            SemanticPosition.SAME_BACK: self.vehicle_current_lane_back,
            SemanticPosition.SAME_FRONT: self.vehicle_current_lane_front,
            SemanticPosition.NEXT_BACK: self.vehicle_next_lane_back,
            SemanticPosition.NEXT_FRONT: self.vehicle_next_lane_front,
            SemanticPosition.EGO: self.ego
        }
