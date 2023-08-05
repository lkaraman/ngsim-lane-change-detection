import sympy as sp
from sympy.utilities.lambdify import lambdify

from structs import Trajectory, SemanticPosition, vehicle_far_far_away, VehicleFrame

# --------------------------------
# Parameters for the planner
# --------------------------------
wgx = 0.5
wgy = 0.0
ws = 0.5
wa = 0.5

theta_ax = 1.0
theta_ay = 1.0
theta_s = 1.0
# --------------------------------

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

    def get_trajectory(self, f1, f2, x, y) -> Trajectory:
        x_l = [x]
        y_l = [y]

        for i in range(100):
            x = x + f1(x, y)
            y = y + f2(x, y)

            x_l.append(x)
            y_l.append(y)

        return Trajectory(x=x_l, y=y_l)

