import numpy


def cosine_similarity(a, b):
    assert type(a) is numpy.ndarray or type(b) is numpy.ndarray
    return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))
