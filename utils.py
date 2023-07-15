from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from DataAnnotator.Utilities.commons import calculate_route_s_and_d

from shapely import LineString


def get_velocity(distance, time) -> list[float]:
    distance = np.array(distance)
    time = np.array(time)
    if not distance.shape == time.shape:
        raise ValueError('Mismatch between time and position dimensions!')
    dist_diff = np.diff(distance)
    time_diff = np.diff(time)
    velocity = np.append(dist_diff[0] / time_diff[0], dist_diff / time_diff)
    return velocity.tolist()


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


def find_timespan_in_ms(df_global) -> int:
    # 5 seems to be the first frame for us101
    time_min, time_max = np.min(df_global.Global_Time.values), np.max(df_global.Global_Time.values)
    return np.arange(time_min, time_max, 100)


def convert_dataframe_to_vehicle(df, road: Road, initial_time) -> Vehicle:
    ids = df.Vehicle_ID.values.tolist()
    assert len(set(ids)) == 1

    time_start = df['Global_Time'].values[0]
    time_end = df['Global_Time'].values[-1]

    frame_start = np.where(initial_time == time_start)[0]
    frame_end = np.where(initial_time == time_end)[0]

    assert len(frame_start) == 1
    assert len(frame_end) == 1

    frame_start = frame_start[0]
    frame_end = frame_end[0]

    x = np.asarray(df.Global_X) / 3.28
    y = np.asarray(df.Global_Y) / 3.28

    r = calculate_route_s_and_d(x_ref=np.array(road.reference_line.x), y_ref=np.array(road.reference_line.y),
                                x=np.asarray([df.Global_X]).flatten() / 3.28,
                                y=np.asarray([df.Global_Y]).flatten() / 3.28, tol=0.1)
    width = df.v_Width / 3.28
    length = df.v_length / 3.28

    times = list((df['Global_Time'].values - initial_time[0]) / 1000)

    times = list(np.arange(0, len(x)) * 0.1 + times[0])

    vel_lon = get_velocity(distance=r[0],
                           time=times)

    vel_lat = get_velocity(distance=r[1],
                           time=times)

    lanes = [current_lane(i, list(road.lanes.values())) for i in r[-1]]

    return Vehicle(
        object_id=ids[0],
        s=r[0].tolist(),
        d=r[1].tolist(),
        frames=list(range(frame_start, frame_end + 1)),
        length=length.values.tolist(),
        width=width.values.tolist(),
        times=times,
        vs=vel_lon,
        vd=vel_lat,
        lanes=lanes
    )


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

def decorate_with_far_vehicles(semantic_frames: SemanticFrames) -> SemanticFrames:
        semantic_frames[SemanticPosition.SAME_BACK] = semantic_frames[SemanticPosition.SAME_BACK] or vehicle_far_far_away
        semantic_frames[SemanticPosition.SAME_FRONT] = semantic_frames[SemanticPosition.SAME_FRONT] or vehicle_far_far_away
        semantic_frames[SemanticPosition.NEXT_BACK] = semantic_frames[SemanticPosition.NEXT_BACK] or vehicle_far_far_away
        semantic_frames[SemanticPosition.NEXT_FRONT] = semantic_frames[SemanticPosition.NEXT_FRONT] or vehicle_far_far_away

        return semantic_frames


def current_lane(d, lane_lines: List[Lane]):
    current_lane = -1

    for i, (l_p, l_n) in enumerate(zip(lane_lines, lane_lines[1:])):
        if d < l_p.y[0] and d > l_n.y[0]:
            return i

    return current_lane


def convert_to_vehicle_frame(ego_id, frame, vehicles: List[Vehicle], lane_lines) -> List[VehicleFrame]:
    vehicle_frames = []

    def is_ego(id, frm, v: Vehicle):
        return v.object_id == id and frm in v.frames

    for v in vehicles:
        if frame not in v.frames:
            continue

        rel_frm = frame - v.frames[0]

        s0 = v.s[rel_frm]
        d0 = v.d[rel_frm]

        lane_id = current_lane(d0, lane_lines=lane_lines)

        veh_frame = VehicleFrame(object_id=v.object_id,
                                 s=s0,
                                 d=d0,
                                 lane=lane_id,
                                 is_ego=is_ego(ego_id, frame, v),
                                 width=v.width[rel_frm],
                                 length=v.length[rel_frm],
                                 velocity=v.vs[rel_frm])

        vehicle_frames.append(veh_frame)

    return vehicle_frames


def convert_df_to_vf(df_frame, ref_x, ref_y, lane_lines, ego_id) -> List[VehicleFrame]:
    vehicle_frames = []

    for _, i in df_frame.iterrows():
        r = calculate_route_s_and_d(x_ref=np.array(ref_x), y_ref=np.array(ref_y),
                                    x=np.asarray([i.Global_X]) / 3.28,
                                    y=np.asarray([i.Global_Y]) / 3.28, tol=0.1)
        lane_id = current_lane(d=r[1][0], lane_lines=lane_lines)

        vf = VehicleFrame(object_id=i.Vehicle_ID,
                          s=r[0][0],
                          d=r[1][0],
                          lane=lane_id,
                          is_ego=ego_id == i.Vehicle_ID)
        vehicle_frames.append(vf)

    return vehicle_frames


def split_vf_for_ego(vfl: List[VehicleFrame]) -> (VehicleFrame, List[VehicleFrame]):
    ego = [i for i in vfl if i.is_ego]
    fellows = [i for i in vfl if not i.is_ego]

    assert len(ego) == 1
    ego = ego[0]

    return ego, fellows


def find_vehicle_in_frame(veh_id: int, veh_frames: list[VehicleFrame]):
    if veh_id == -1:
        return None
    l = [vf for vf in veh_frames if vf.object_id == veh_id]
    assert len(l) == 1

    return l[0]


def find_id_by_semantic_position(ego: VehicleFrame, fellows: List[VehicleFrame], semantic: SemanticPosition) -> int:
    ego_s = ego.s
    ego_lane = ego.lane

    if semantic == SemanticPosition.SAME_BACK:
        condition_lat = lambda f: ego_lane == f.lane
        condition_long = lambda f: ego_s > f.s

    elif semantic == SemanticPosition.SAME_FRONT:
        condition_lat = lambda f: ego_lane == f.lane
        condition_long = lambda f: ego_s < f.s

    elif semantic == SemanticPosition.NEXT_BACK:
        condition_lat = lambda f: ego_lane == (f.lane + 1)
        condition_long = lambda f: ego_s > f.s

    elif semantic == SemanticPosition.NEXT_FRONT:
        condition_lat = lambda f: ego_lane == (f.lane + 1)
        condition_long = lambda f: ego_s < f.s

    else:
        raise AttributeError('Semantic position not valid!')

    rel_s = []
    for f in fellows:
        if condition_lat(f) and condition_long(f):
            rel_s.append((f.object_id, abs(ego_s - f.s)))

    if len(rel_s) == 0:
        return -1

    m = min(rel_s, key=lambda x: x[1])
    return m[0]


@dataclass
class AnnotationEntry:
    vehicle_id: int
    index_start: int
    index_end: int


def get_surrounding_vehicles_frames(ego_id, frame, vehicles, lane_lines):
    vehicle_frame = convert_to_vehicle_frame(ego_id=ego_id,
                                             frame=frame,
                                             vehicles=vehicles,
                                             lane_lines=lane_lines)

    ego, fellows = split_vf_for_ego(vehicle_frame)

    back_id = find_id_by_semantic_position(ego, fellows, semantic=SemanticPosition.SAME_BACK)
    front_id = find_id_by_semantic_position(ego, fellows, semantic=SemanticPosition.SAME_FRONT)
    left_back_id = find_id_by_semantic_position(ego, fellows, semantic=SemanticPosition.NEXT_BACK)
    left_front_id = find_id_by_semantic_position(ego, fellows, semantic=SemanticPosition.NEXT_FRONT)

    back = find_vehicle_in_frame(veh_id=back_id, veh_frames=vehicle_frame)
    front = find_vehicle_in_frame(veh_id=front_id, veh_frames=vehicle_frame)
    left_back = find_vehicle_in_frame(veh_id=left_back_id, veh_frames=vehicle_frame)
    left_front = find_vehicle_in_frame(veh_id=left_front_id, veh_frames=vehicle_frame)
    return SurroundingVehicleFrames(vehicle_current_lane_front=front,
                                    vehicle_current_lane_back=back,
                                    vehicle_next_lane_front=left_front,
                                    vehicle_next_lane_back=left_back,
                                    ego=ego
                                    )
