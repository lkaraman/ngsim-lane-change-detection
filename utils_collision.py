from typing import Union, Sequence

import numpy as np
Vector = Union[np.ndarray, Sequence[float]]

def rotated_rectangles_intersect(rect1: tuple[Vector, float, float, float],
                                 rect2: tuple[Vector, float, float, float]) -> bool:
    """
    Do two rotated rectangles intersect?
    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    :return: do they?
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)


def has_corner_inside(rect1: tuple[Vector, float, float, float],
                      rect2: tuple[Vector, float, float, float]) -> bool:
    """
    Check if rect1 has a corner inside rect2
    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    (c1, l1, w1, a1) = rect1
    (c2, l2, w2, a2) = rect2
    c1 = np.array(c1)
    l2v = np.array([l2 / 2, 0])
    w2v = np.array([0, w2 / 2])

    r2_points = np.array([[0, 0],
                          - l2v, l2v, -w2v, w2v,
                          - l2v - w2v, - l2v + w2v, + l2v - w2v, + l2v + w2v])

    c, s = np.cos(a2), np.sin(a2)
    r2 = np.array([[c, -s], [s, c]])

    c, s = np.cos(-a1), np.sin(-a1)
    r3 = np.array([[c, -s], [s, c]])

    rotated_r2_points = r2.dot(r2_points.transpose()).transpose() + [c2[0], c2[1]] - [c1[0], c1[1]]
    rotated_r2_points = r3.dot(rotated_r2_points.transpose()).transpose()

    return any([point_in_rectangle(p, (-l1 / 2, -w1 / 2), (l1 / 2, w1 / 2)) for p in rotated_r2_points])


def point_in_rectangle(point: Vector, rect_min: Vector, rect_max: Vector) -> bool:
    """
    Check if a point is inside a rectangle
    :param point: a point (x, y)
    :param rect_min: x_min, y_min
    :param rect_max: x_max, y_max
    """
    return rect_min[0] <= point[0] <= rect_max[0] and rect_min[1] <= point[1] <= rect_max[1]