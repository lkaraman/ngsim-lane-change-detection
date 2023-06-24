from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

from DataAnnotator.Utilities.commons import calculate_route_s_and_d


def calculate_velocity(driven_distance, time):
    """
    Calculates the velocities for every frame with regard to the driven distance.
    :param driven_distance: 1-dim vector of the driven distances (cumulated)
    :param time: 1-dim time vector
    :return: 1-dim velocity vector
    """
    driven_distance = np.asarray(driven_distance)
    time = np.asarray(time)
    if not driven_distance.shape == time.shape:
        raise ValueError('The dimensions of the arguments are not matching')
    dist_diff = np.diff(driven_distance)
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


    vel_lon = calculate_velocity(driven_distance=r[0],
                                 time=times)

    vel_lat = calculate_velocity(driven_distance=r[1],
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

        rel_frm = frame-v.frames[0]

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


def find_back_id(ego: VehicleFrame, fellows: List[VehicleFrame]) -> int:
    ego_s = ego.s
    ego_lane = ego.lane

    rel_s = []
    for f in fellows:
        if ego_lane == f.lane and ego_s > f.s:
            rel_s.append((f.object_id, abs(ego_s - f.s)))

    if len(rel_s) == 0:
        return -1

    m = min(rel_s, key=lambda x: x[1])
    return m[0]


def find_front_id(ego: VehicleFrame, fellows: List[VehicleFrame]) -> int:
    ego_s = ego.s
    ego_lane = ego.lane

    rel_s = []
    for f in fellows:
        if ego_lane == f.lane and ego_s < f.s:
            rel_s.append((f.object_id, abs(ego_s - f.s)))

    if len(rel_s) == 0:
        return -1

    m = min(rel_s, key=lambda x: x[1])
    return m[0]

def find_left_front_id(ego: VehicleFrame, fellows: List[VehicleFrame]) -> int:
    ego_s = ego.s
    ego_lane = ego.lane

    rel_s = []
    for f in fellows:
        if ego_lane == (f.lane+1) and ego_s < f.s:
            rel_s.append((f.object_id, abs(ego_s - f.s)))

    if len(rel_s) == 0:
        return -1

    m = min(rel_s, key=lambda x: x[1])
    return m[0]

def find_left_back_id(ego: VehicleFrame, fellows: List[VehicleFrame]) -> int:
    ego_s = ego.s
    ego_lane = ego.lane

    rel_s = []
    for f in fellows:
        if ego_lane == (f.lane + 1) and ego_s > f.s:
            rel_s.append((f.object_id, abs(ego_s - f.s)))

    if len(rel_s) == 0:
        return -1

    m = min(rel_s, key=lambda x: x[1])
    return m[0]

def find_vehicle_in_frame(veh_id: int, veh_frames: list[VehicleFrame]):
    l = [vf for vf in veh_frames if vf.object_id == veh_id]
    assert len(l) == 1

    return l[0]


@dataclass
class AnnotationEntry:
    vehicle_id: int
    index_start: int
    index_end: int