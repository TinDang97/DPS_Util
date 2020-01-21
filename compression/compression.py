import json
import numpy
import zlib
import pickle

SUPPORT_TYPE = [numpy.ndarray, dict, bytes, str, list, int, float]


def compress(data, compress_lv=lzma.PRESET_EXTREME, check=lzma.CHECK_SHA256):
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
    compressed = lzma.compress(data, preset=compress_lv, check=check, format=lzma.FORMAT_XZ)
    return compressed


def decompress(binary):
    data = lzma.decompress(binary, format=lzma.FORMAT_XZ)
    try:
        data = json.loads(data)
    except json.JSONDecodeError and UnicodeDecodeError:
        data = pickle.loads(data)
    return data
