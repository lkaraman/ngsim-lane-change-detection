from unittest.mock import Mock

from feature_generator import FeatureGenerator
from utils import Vehicle, Traffic, AnnotationEntry

import numpy.testing


def test_feature_generator() -> None:


    vehicle = Vehicle(object_id=55,
                      s=[10, 20, 30],
                      d=[4, 4, 4],
                      times=[10, 10.5, 11],
                      frames=[5, 6, 7],
                      length=[4.0, 4.0, 4.0],
                      width=[2.0, 2.0, 2.0],
                      vs=[1, 1, 1],
                      vd=[5, 5, 5],
                      lanes=[1, 1, 1])
    traffic = Traffic(vehicles=[vehicle])

    road_helper = Mock()
    road_helper.closest_distance_to_lane_line.return_value = [1.0, 2.0, 3.0]
    road_helper.road.lanes = {0: [3.5]}

    feature_generator = FeatureGenerator(traffic=traffic,
                                         window_size=2,
                                         road_helper=road_helper,
                                         use_potential_feature=False)

    annotation_entry = AnnotationEntry(vehicle_id=55,
                                       index_start=6,
                                       index_end=7)

    inn, out = feature_generator.create_input_and_output_matrices(annotations=[annotation_entry])

    numpy.testing.assert_almost_equal(inn, [[1, 1, 2, 0, 5, 5], [1, 2, 3, 5, 5, 5], [2, 3, 3, 5, 5, 0]])
    numpy.testing.assert_almost_equal(out, [[0], [1], [1]])

    pass
