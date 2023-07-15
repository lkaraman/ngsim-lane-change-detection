from typing import List

import numpy as np

from DataAnnotator.Utilities.commons import calculate_route_s_and_d
from structs import Vehicle, Lane, SemanticPosition, SurroundingVehicleFrames, SemanticFrames, \
    vehicle_far_far_away, VehicleFrame


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
