import json

from dacite import from_dict

from feature_generator import FeatureGenerator
from road_helper import RoadHelper
from structs import Scenario, AnnotationEntry

if __name__ == '__main__':
    with open('input/scenario.json', 'r') as f:
        d = json.load(f)

    scenario = from_dict(Scenario, d)

    with open('input/lc_anno.json', 'r') as f:

        annos: list[AnnotationEntry] = []

        for line in f:
            t = line.replace(' ', '')
            t = t.split('-')
            assert len(t) == 3

            annos.append(
                AnnotationEntry(
                    vehicle_id=int(t[0]),
                    index_start=int(t[1]),
                    index_end=int(t[2])
                )
            )


    road_helper = RoadHelper(road=scenario.road)
    fg = FeatureGenerator(traffic=scenario.traffic,
                          window_size=50,
                          road_helper=road_helper)

    fg.train(annotations=annos)
    fg.predict()