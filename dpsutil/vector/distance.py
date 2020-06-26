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
    assert x1.shape[-1] == x2.shape[-1]
    assert len(x1.shape) <= 2

    if not skip_normalize:
        if len(x1.shape) == 2:
            x1 = normalize_L2(x1)
            x2 = normalize_L2(x2)
        else:
            x1 = normalize_L1(x1)
            x2 = normalize_L1(x2)

    return numpy.dot(x1, x2.T)


inner_product = (lambda x1, x2: cosine_similarity(x1, x2, True))
cosine = (lambda x1, x2, skip_normalize=False: cosim2cosine(cosine_similarity(x1, x2, skip_normalize=skip_normalize)))


def euclidean_distance(x1, x2):
    x1 = numpy.asarray(x1)
    x2 = numpy.asarray(x2)

    assert type(x1) is numpy.ndarray or type(x2) is numpy.ndarray
    assert x1.shape == x2.shape
    assert len(x1.shape) <= 2

    if len(x1.shape) == 1:
        return numpy.sqrt(numpy.sum((x1 - x2) ** 2))

    return numpy.sqrt(numpy.sum((x1[:, numpy.newaxis, :] - x2[numpy.newaxis, :, :]) ** 2, axis=-1))


def absolute_distance(x1, x2):
    return numpy.sum(numpy.absolute(x1 - x2))


l2_distance = euclidean_distance
cosine2cosim = (lambda cos: 1 - cos)
cosim2cosine = (lambda cosim: cosine2cosim(cosim))


def cosine2euclid(cos):
    """
    Convert cosine -> normalized euclidean distance

    Note: this's lossy method.
    """
    return numpy.sqrt(2 * cos)


cosim2euclid = (lambda cosim: cosine2euclid(cosim2cosine(cosim)))


def euclid2cosine(euclid_dis):
    """
    Convert normalized euclidean distance -> cosine
    Note: this's lossy method.
    """
    return (euclid_dis ** 2) / 2


euclid2cosim = (lambda euclid_dis: cosine2cosim(euclid2cosine(euclid_dis)))


__all__ = ['normalize_L1', 'normalize_L2',
           'cosine_similarity', 'cosine', 'euclidean_distance', 'l2_distance', 'absolute_distance', 'inner_product',
           'cosim2euclid', 'euclid2cosim', 'euclid2cosine', 'cosine2cosim', 'cosim2cosine', 'cosine2euclid']
