import json

from dacite import from_dict

from feature_generator import FeatureGenerator
from road_helper import RoadHelper
from structs import Scenario

if __name__ == '__main__':
    with open('/home/luka/WeekendProjects/ngim_lane_change_detection/DataAnnotator/scenario.json', 'r') as f:
        d = json.load(f)

    scenario = from_dict(Scenario, d)


    road_helper = RoadHelper(road=scenario.road)
    fg = FeatureGenerator(traffic=scenario.traffic,
                          window_size=50,
                          road_helper=road_helper)

    fg.try_out()