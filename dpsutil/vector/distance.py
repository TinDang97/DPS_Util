import numpy


def normalize_L1(x):
    return x / numpy.linalg.norm(x)


def normalize_L2(x):
    assert isinstance(x, numpy.ndarray)
    return x / numpy.sqrt(numpy.sum((x ** 2), keepdims=True, axis=1))


def cosine_similarity(x1, x2, skip_normalize=False):
    if type(x1) is list:
        x1 = numpy.array(x1)

    if type(x2) is list:
        x2 = numpy.array(x2)

    assert type(x1) is numpy.ndarray or type(x2) is numpy.ndarray
    assert x1.shape == x2.shape
    assert len(x1.shape) <= 2

    if not skip_normalize:
        if len(x1.shape) == 2:
            x1 = normalize_L2(x1)
            x2 = normalize_L2(x2)
        else:
            x1 = normalize_L1(x1)
            x2 = normalize_L1(x2)

    return numpy.dot(x1, x2.T)


def cosine(x1, x2, skip_normalize=False):
    return 1 - cosine_similarity(x1, x2, skip_normalize=skip_normalize)


def euclidean_distance(x1, x2):
    if type(x1) is list:
        x1 = numpy.array(x1)

    if type(x2) is list:
        x2 = numpy.array(x2)

    assert type(x1) is numpy.ndarray or type(x2) is numpy.ndarray
    assert x1.shape == x2.shape
    assert len(x1.shape) <= 2

    if len(x1.shape) == 1:
        return numpy.sqrt(numpy.sum((x1 - x2) ** 2))

    return numpy.sqrt(numpy.sum((x1[:, numpy.newaxis, :] - x2[numpy.newaxis, :, :]) ** 2, axis=-1))


def cosim2euclid(cosim):
    """
    Convert cosine similarity -> normalized euclidean distance
    :return:
    """
    return numpy.sqrt(cosim)


def euclid2cosim(euclid_dis):
    """
    Convert normalized euclidean distance -> cosine similarity
    :return:
    """
    return euclid_dis ** 2


def absolute_distance(x1, x2):
    return numpy.sum(numpy.absolute(x1 - x2))


__all__ = ['normalize_L1', 'normalize_L2', 'cosine_similarity', 'cosine', 'euclidean_distance',
           'cosim2euclid', 'euclid2cosim', 'absolute_distance']
