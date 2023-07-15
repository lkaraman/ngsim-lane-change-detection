from dataclasses import dataclass
from typing import List, Dict

import matplotlib.pyplot as plt


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