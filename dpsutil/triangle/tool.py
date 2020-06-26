import numpy


def cos2degree(cos) -> float:
    """
    Convert cosine to degree. Support float, List[float], numpy.ndarray[float]
    :param cos:
    :return:
    """
    return numpy.arccos(cos) / numpy.pi * 180


def cos_angle(left_side: float, right_side: float, third_side: float) -> float:
    """
    Calculate cos of angle if give 3 side of triangle
    """
    return (left_side ** 2 + right_side ** 2 - third_side ** 2) / (2 * left_side * right_side)


def degree_angle(left_side: float, right_side: float, third_side: float) -> float:
    """
    Calculate degree of angle if give 3 side of triangle
    """
    return cos2degree(cos_angle(left_side, right_side, third_side))


def cos_triangle(first_side, second_side, third_side):
    """
    Calculate 3 angles's cos of triangle
    """
    first_angle = cos_angle(second_side, third_side, first_side)
    second_angle = cos_angle(first_side, third_side, second_side)
    third_angle = cos_angle(first_side, second_side, third_side)
    return first_angle, second_angle, third_angle


def degree_triangle(first_side, second_side, third_side):
    return cos2degree(cos_triangle(first_side, second_side, third_side))
