import json

import blosc
import numpy
import zlib
import pickle

SUPPORT_TYPE = [numpy.ndarray, dict, bytes, str, list, int, float]


def compress(data):
    """
    if ndarray => using numpy tobytes, otherwise using json.dumps -> encode("utf-8")
    :param check:
    :param compress_lv:
    :param data:
    :return:
    """
    assert type(data) in SUPPORT_TYPE
    if isinstance(data, numpy.ndarray):
        data = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
    if not isinstance(data, bytes):
        data = json.dumps(data).encode("utf-8")
    compressed = zlib.compress(data)
    return compressed


def decompress(binary):
    data = zlib.decompress(binary)
    try:
        data = json.loads(data)
    except json.JSONDecodeError and UnicodeDecodeError:
        data = pickle.loads(data)
    return data


def compress_ndarray(vectors):
    assert isinstance(vectors, numpy.ndarray)
    return pickle.dumps([blosc.compress(vectors, clevel=5, cname="lz4"), vectors.dtype, vectors.shape])


def decompress_ndarray(binary):
    buffer, dtype, shape = pickle.loads(binary)
    return numpy.frombuffer(blosc.decompress(buffer), dtype=dtype).reshape(shape)