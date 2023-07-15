import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify

from trajectory_predictor import TrajectoryPredictor
from utils import VehicleFrame, Trajectory, SemanticPosition, vehicle_far_far_away
from visualize_predicted import PredictVisualizer

wgx = 0.5
wgy = 0.0
ws = 0.5
wa = 0.5

theta_ax = 1.0
theta_ay = 1.0
theta_s = 1.0

def gradient(f):
    return (f.diff(x), f.diff(y))


class TrajectoryPlanner:

    def __init__(self):
        self.x = sp.symbols('x')
        self.y = sp.symbols('y')

        self.x_other = sp.symbols('xf0:4')
        self.y_other = sp.symbols('yf0:4')

        self.l_up = sp.symbols('lu')
        self.l_down = sp.symbols('ld')

        self.dU = self.get_der_function()

    def compute_ua(self):
        res = 0
        for i in range(4):
            res += wa * sp.exp(
                -((self.x - self.x_other[i]) ** 2) / theta_ax ** 2 - ((self.y - self.y_other[i]) ** 2) / theta_ay ** 2)

        return res

    def compute_ug(self):
        return -wgx * self.x - wgy * self.y

    def compute_us(self):
        return ws * (-sp.exp(-(self.y - self.l_up) ** 2 / theta_s ** 2) + sp.exp(
            -(self.y - self.l_down) ** 2 / theta_s ** 2))

    def get_der_function(self):
        Ug = self.compute_ug()
        Us = self.compute_us()
        Ua = self.compute_ua()

        U = Ug + Us + Ua

        res = (-U.diff(self.x), -U.diff(self.y))

        return res

    def get_field_functions(self, surrounding_vehicles_frame: dict[SemanticPosition, VehicleFrame]):
        g1 = self.dU[0].subs([
            [self.l_up, 3.5],
            [self.l_down, 0],
            [self.x_other[0], (surrounding_vehicles_frame[SemanticPosition.SAME_BACK] or vehicle_far_far_away).s],
            [self.y_other[0], (surrounding_vehicles_frame[SemanticPosition.SAME_BACK] or vehicle_far_far_away).d],
            [self.x_other[1], (surrounding_vehicles_frame[SemanticPosition.SAME_FRONT] or vehicle_far_far_away).s],
            [self.y_other[1], (surrounding_vehicles_frame[SemanticPosition.SAME_FRONT] or vehicle_far_far_away).d],
            [self.x_other[2], (surrounding_vehicles_frame[SemanticPosition.NEXT_BACK] or vehicle_far_far_away).s],
            [self.y_other[2], (surrounding_vehicles_frame[SemanticPosition.NEXT_BACK] or vehicle_far_far_away).d],
            [self.x_other[3], (surrounding_vehicles_frame[SemanticPosition.NEXT_FRONT] or vehicle_far_far_away).s],
            [self.y_other[3], (surrounding_vehicles_frame[SemanticPosition.NEXT_FRONT] or vehicle_far_far_away).d],
        ])

        g2 = self.dU[1].subs([
            [self.l_up, 3.5],
            [self.l_down, 0],
            [self.x_other[0], (surrounding_vehicles_frame[SemanticPosition.SAME_BACK] or vehicle_far_far_away).s],
            [self.y_other[0], (surrounding_vehicles_frame[SemanticPosition.SAME_BACK] or vehicle_far_far_away).d],
            [self.x_other[1], (surrounding_vehicles_frame[SemanticPosition.SAME_FRONT] or vehicle_far_far_away).s],
            [self.y_other[1], (surrounding_vehicles_frame[SemanticPosition.SAME_FRONT] or vehicle_far_far_away).d],
            [self.x_other[2], (surrounding_vehicles_frame[SemanticPosition.NEXT_BACK] or vehicle_far_far_away).s],
            [self.y_other[2], (surrounding_vehicles_frame[SemanticPosition.NEXT_BACK] or vehicle_far_far_away).d],
            [self.x_other[3], (surrounding_vehicles_frame[SemanticPosition.NEXT_FRONT] or vehicle_far_far_away).s],
            [self.y_other[3], (surrounding_vehicles_frame[SemanticPosition.NEXT_FRONT] or vehicle_far_far_away).d],
        ])


        f1 = lambdify([self.x, self.y], g1)
        f2 = lambdify([self.x, self.y], g2)

        return f1, f2


if __name__ == '__main__':
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

    x = ego_frame.s
    y = ego_frame.d

    x_l = [x]
    y_l = [y]

    for i in range(100):
        x = x + f1(x, y)
        y = y + f2(x, y)

        x_l.append(x)
        y_l.append(y)

    X, Y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-5, 5, 20))

    U = [f1(x1, y1) for x1, y1 in zip(X, Y)]
    V = [f2(x1, y1) for x1, y1 in zip(X, Y)]

    plt.quiver(X, Y, U, V, linewidth=1)
    plt.scatter(x_l, y_l)
    plt.title("vector field")
    plt.show()

    tp = TrajectoryPredictor(relevant_frames=surrounding_vehicle_frame,
                             trajectory=Trajectory(x=x_l, y=y_l))

    for i in np.arange(0.1, 4, 0.1):

        tp.predict_for_dt(dt=i)
        tp.is_collision_in_predicted()

    print(tp.collision_info)

    vis = PredictVisualizer(tp.predicted_states, gradient_fnc=(f1, f2), traj=(x_l, y_l))






    pass