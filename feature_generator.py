import numpy as np
from scipy.special import kn
from sklearn import svm

from road_helper import RoadHelper
from utils import Scenario, Vehicle, AnnotationEntry, convert_to_vehicle_frame, split_vf_for_ego, find_back_id, \
    find_front_id, find_left_back_id, find_left_front_id, find_vehicle_in_frame, Traffic


class FeatureGenerator:
    def __init__(self, traffic: Traffic, window_size: int,  road_helper: RoadHelper, use_potential_feature: bool = True,) -> None:
        if window_size % 2 != 0:
            raise ValueError('Window size must be even')

        self.window_size = window_size
        self.use_potential_feature = use_potential_feature
        self.traffic = traffic

        self.road_helper = road_helper

        self.model = None

    def _compute_output_matrix_for_vehicle_annotation(self, vehicle_annotation: AnnotationEntry) -> np.ndarray:
        vehicle = self.traffic.find_vehicle_with_id_and_containing_frames(veh_id=vehicle_annotation.vehicle_id,
                                                                                   frame_s=vehicle_annotation.index_start,
                                                                                   frame_end=vehicle_annotation.index_end)
        Y = np.empty((0, 1))

        for f in range(len(vehicle.frames)):
            y = self.get_y_for_vehicle(veh_anno=vehicle_annotation, global_frame=vehicle.frames[0] + f)
            Y = np.vstack((Y, y))

        return Y


    def _append_potential_feature(self, to_pad, X: np.ndarray, vehicle: Vehicle) -> np.ndarray:
        potentials = [self.get_potential_feature(vehicle.object_id, f) for f in vehicle.frames]

        if np.any(np.isnan(potentials)):
            raise ValueError

        potentials_with_ghost_frames = np.hstack((np.repeat(potentials[0], to_pad),
                                                  potentials,
                                                  np.repeat(potentials[-1], to_pad)))

        X_potential = np.empty((0, 1 * (self.window_size + 1)))

        for f in range(len(vehicle.frames)):
            x_pot = potentials_with_ghost_frames[f: f + self.window_size + 1]
            X_potential = np.vstack((X_potential, x_pot))


        return np.hstack((X, X_potential))




    def _compute_input_matrix_for_vehicle_with_potential(self, vehicle: Vehicle) -> np.ndarray:
        closest_distance_to_line_line = self.road_helper.closest_distance_to_lane_line(s=vehicle.s,
                                                                                       d=vehicle.d)

        lateral_velocity = vehicle.vd

        # For ghost frames we assume the closest distance is the same and lateral velocity is zero
        to_padd = self.window_size / 2



        closest_distance_start = closest_distance_to_line_line[0]
        closest_distance_end = closest_distance_to_line_line[-1]
        closest_distance_to_line_line_with_ghost_frames = np.hstack((np.repeat(closest_distance_start, to_padd),
                                                                     closest_distance_to_line_line,
                                                                     np.repeat(closest_distance_end, to_padd)))
        lateral_velocity_with_ghost_frames = np.hstack((np.repeat(0, to_padd),
                                                        lateral_velocity,
                                                        np.repeat(0, to_padd)))



        X = np.empty((0, 2 * (self.window_size + 1)))

        for f in range(len(vehicle.frames)):
            x_dist = closest_distance_to_line_line_with_ghost_frames[f:f + self.window_size + 1]
            x_vel = lateral_velocity_with_ghost_frames[f:f + self.window_size + 1]

            x = np.hstack((x_dist, x_vel))
            X = np.vstack((X, x))

        if self.use_potential_feature is False:
            return X

        return self._append_potential_feature(to_pad=to_padd,
                                              vehicle=vehicle,
                                              X=X)

    def get_potential_feature(self, ego_id: int, frame: int) -> float:
        vehicle_frame = convert_to_vehicle_frame(ego_id=ego_id,
                                                 frame=frame,
                                                 vehicles=self.traffic.vehicles,
                                                 lane_lines=list(self.road_helper.road.lanes.values()))

        ego, fellows = split_vf_for_ego(vehicle_frame)

        back_id = find_back_id(ego, fellows)
        front_id = find_front_id(ego, fellows)
        left_back_id = find_left_back_id(ego, fellows)
        left_front_id = find_left_front_id(ego, fellows)

        vehs = (back_id, front_id, left_back_id, left_front_id)

        pot = []

        eta = 0.5
        wp = 0.5
        wf = 0.5
        wl = 0.5
        wr = 0.5
        dev = 1

        for i, v in enumerate(vehs):
            if v != -1:
                rel_v = find_vehicle_in_frame(veh_id=v, veh_frames=vehicle_frame)
                relative_distance = abs(rel_v.s - ego.s)
                relative_velocity = abs(rel_v.velocity - ego.velocity)
            else:
                relative_distance = 50
                relative_velocity = 0.0

            U = np.exp(eta * relative_velocity) / (2 * np.pi * kn(0, eta * relative_velocity)) * np.exp(
                -relative_distance ** 2 / 2) / (2 * np.pi)

            pot.append(U)

        Uc = wp * pot[1] + wf * pot[0]
        Un = wl * pot[3] + wr * pot[2]

        Uc = np.clip(Uc, 0.1, 1)
        Un = np.clip(Un, 0.1, 1)

        return np.log(Uc) - np.log(Un)

    def create_input_and_output_matrices(self, annotations: list[AnnotationEntry]):
        if self.use_potential_feature is True:
            XX = np.empty((0, 3 * (self.window_size + 1)))
        else:
            XX = np.empty((0, 2 * (self.window_size + 1)))

        YY = np.empty((0, 1))

        for v in annotations:
            try:
                vehicle = self.traffic.find_vehicle_with_id_and_containing_frames(veh_id=v.vehicle_id,
                                                                                           frame_s=v.index_start,
                                                                                           frame_end=v.index_end)

                X = self._compute_input_matrix_for_vehicle_with_potential(vehicle=vehicle)
                Y = self._compute_output_matrix_for_vehicle_annotation(vehicle_annotation=v)
            except AssertionError:
                continue

            XX = np.vstack((XX, X))
            YY = np.vstack((YY, Y))

        return XX, YY

    def ff(self, vv: list[AnnotationEntry]):
        # XX = np.empty((0, 2 * (self.window_size + 1)))
        XX = np.empty((0, 3 * (self.window_size + 1)))

        YY = np.empty((0, 1))

        for v in vv:
            try:
                vehicle = self.traffic.find_vehicle_with_id_and_containing_frames(veh_id=v.vehicle_id,
                                                                                           frame_s=v.index_start,
                                                                                           frame_end=v.index_end)

                X = self._compute_input_matrix_for_vehicle_with_potential(vehicle=vehicle)
                Y = self._compute_output_matrix_for_vehicle_annotation(vehicle_annotation=v)
            except AssertionError:
                continue

            XX = np.vstack((XX, X))
            YY = np.vstack((YY, Y))

        clf = svm.SVC()
        clf.fit(XX, YY)

        self.model = clf

        vehicle_to_test = self.traffic.vehicles[-4]

        for vehicle_to_test in self.traffic.vehicles[1000:1050]:
            x_to_test = self._compute_input_matrix_for_vehicle_with_potential(vehicle=vehicle_to_test)

            a = clf.predict(x_to_test)
            print(a)

            nnn = np.where(a == 1)[0]

            import matplotlib.pyplot as plt

            for r in self.road_helper.road.lanes.values():
                plt.plot(r.x, r.y, c='b')

            plt.plot(vehicle_to_test.s, vehicle_to_test.d)
            for n in nnn:
                plt.scatter(vehicle_to_test.s[n], vehicle_to_test.d[n], c='r')
            plt.show()

    def get_y_for_vehicle(self, veh_anno: AnnotationEntry, global_frame: int):
        if veh_anno.index_start <= global_frame <= veh_anno.index_end:
            return [1]
        return [0]
