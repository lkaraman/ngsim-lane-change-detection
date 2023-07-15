import pytest

from trajectory_planner import TrajectoryPlanner
from utils import VehicleFrame, SemanticPosition


def test_trajectory_predictor():
    tp = TrajectoryPlanner()

    vehicle_current_lane_back = VehicleFrame(
        object_id=0,
        s=-5,
        d=1.75,
        lane=0,
        is_ego=False,
        width=2,
        length=2,
        velocity=20
    )

    vehicle_current_lane_front = VehicleFrame(
        object_id=1,
        s=5,
        d=1.75,
        lane=0,
        is_ego=False,
        width=2,
        length=2,
        velocity=20
    )

    vehicle_next_lane_back = VehicleFrame(
        object_id=2,
        s=-7,
        d=-1.75,
        lane=0,
        is_ego=False,
        width=2,
        length=2,
        velocity=30
    )

    vehicle_next_lane_front = VehicleFrame(
        object_id=3,
        s=3,
        d=-1.75,
        lane=0,
        is_ego=False,
        width=2,
        length=2,
        velocity=20
    )

    ego_frame = VehicleFrame(
        object_id=99,
        s=0,
        d=0,
        lane=0,
        is_ego=True,
        width=2,
        length=4,
        velocity=20
    )

    surrounding_vehicle_frame = {
        SemanticPosition.SAME_BACK: vehicle_current_lane_back,
        SemanticPosition.SAME_FRONT: vehicle_current_lane_front,
        SemanticPosition.NEXT_BACK: vehicle_next_lane_back,
        SemanticPosition.NEXT_FRONT: vehicle_next_lane_front,
        SemanticPosition.EGO: ego_frame
    }

    f1, f2 = tp.get_field_functions(surrounding_vehicles_frame=surrounding_vehicle_frame)

    assert f1(0, 0) == pytest.approx(0.5, abs=0.1)
    assert f2(0, 0) == pytest.approx(0.0, abs=0.1)

    assert f1(4.2, 1.4) == pytest.approx(0.2, abs=0.1)
    assert f2(4.2, 1.4) == pytest.approx(0.1, abs=0.1)

    assert f1(8.9, 2.24) == pytest.approx(0.5, abs=0.1)
    assert f2(8.9, 2.24) == pytest.approx(0.27, abs=0.1)
