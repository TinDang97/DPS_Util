import numpy


def normalize_L2(x):
    assert isinstance(x, numpy.ndarray)
    return x / numpy.sqrt((x ** 2).sum(keepdims=True, axis=1))


def cosine_similarity(a, b):
    assert type(a) is numpy.ndarray or type(b) is numpy.ndarray
    return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))
