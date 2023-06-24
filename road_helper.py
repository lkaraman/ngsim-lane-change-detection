from utils import Road
from shapely.geometry import Point, LineString, MultiLineString


class RoadHelper:

    def __init__(self, road: Road) -> None:
        self.road_shp = list(road.lanes.values())
        self._initialize_shapely_road_representation()



    def _initialize_shapely_road_representation(self) -> None:
        shps: list[LineString] = []

        for r in self.road_shp:
            shps.append(
                LineString(zip(r.x, r.y))
            )

        self.shp_road = MultiLineString(shps)


    def closest_distance_to_lane_line(self, s, d):
        closest_distances = []
        for i, j in zip(s, d):
            closest_distances.append(
                self.shp_road.distance(Point(i, j))
            )

        return closest_distances
