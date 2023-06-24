import sympy as sp
from sympy.tensor.array import derive_by_array
import matplotlib.pyplot as plt
import numpy as np
from sympy.utilities.lambdify import lambdify

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


if __name__ == '__main__':
    tp = TrajectoryPlanner()
    dU = tp.get_der_function()


    g1 = dU[0].subs([
        [tp.l_up, 3.5],
        [tp.l_down, 0],
        [tp.x_other[0], -5],
        [tp.y_other[0], 1.75],
        [tp.x_other[1], 5],
        [tp.y_other[1], 1.75],
        [tp.x_other[2], -7],
        [tp.y_other[2], -1.75],
        [tp.x_other[3], 3],
        [tp.y_other[3], -1.75],
    ])

    g2 = dU[1].subs([
        [tp.l_up, 3.5],
        [tp.l_down, 0],
        [tp.x_other[0], -5],
        [tp.y_other[0], 1.75],
        [tp.x_other[1], 5],
        [tp.y_other[1], 1.75],
        [tp.x_other[2], -7],
        [tp.y_other[2], -1.75],
        [tp.x_other[3], 3],
        [tp.y_other[3], -1.75],
    ])

    x = 0
    y = 0

    x_l = [x]
    y_l = [y]



    X, Y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-5, 5, 20))

    f1 = lambdify([tp.x, tp.y], g1)
    f2 = lambdify([tp.x, tp.y], g2)

    for i in range(25):
        x = x + f1(x, y)
        y = y + f2(x, y)

        x_l.append(x)
        y_l.append(y)

    U = [f1(x1, y1) for x1, y1 in zip(X, Y)]
    V = [f2(x1, y1) for x1, y1 in zip(X, Y)]

    plt.quiver(X, Y, U, V, linewidth=1)
    plt.scatter(x_l, y_l)
    plt.title("vector field")
    plt.show()

    pass