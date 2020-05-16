import pickle
import blosc
import numpy

COMPRESS_FASTEST = 0
COMPRESS_BEST = 1


def compress(data: bytes, compress_type=COMPRESS_FASTEST, nthreads=blosc.ncores, level=None) -> bytes:
    """
    compress(data[, compress_type=COMPRESS_FASTEST, nthreads=blosc.ncores, level=None])
    Optionals:
        - compress_type: [COMPRESS_FASTEST, COMPRESS_BEST]
        - nthreads: range 0 -> 256. Default is the number of cores in this system.
        - level: 0-16. If 'level' is None, compress_type will set.
        Higher values will result in better compression at the cost of more CPU usage.

    High speed compress with multi-threading. Implement from blosc.compress
    Raise ValueError if size of buffer larger than 2147483631 bytes.
    """
    assert type(data) is bytes
    blosc.set_nthreads(nthreads)

    compressor = "lz4" if compress_type == COMPRESS_FASTEST else "zstd"

    if level is None:
        level = 1 if compress_type == COMPRESS_FASTEST else 5
    return blosc.compress(data, cname=compressor, clevel=level)


def decompress(data) -> bytes:
    """
    decompress(data: bytes)
    """
    return blosc.decompress(data)


def compress_ndarray(vectors, compress_type=COMPRESS_FASTEST, nthreads=blosc.ncores) -> bytes:
    """
    compress_ndarray(vectors[, compress_type=COMPRESS_FASTEST, nthreads=blosc.ncores])

    High speed compress numpy.ndarray with multi-threading. Implement from blosc.compress

    Raise ValueError if size of array larger than 2147483631 bytes.
    Example: array with float32 have itemsize=4 and size=614400000 ((1200000, 512) at 2D array)
    -> total size of array: 4*614400000 == 2457600000 bytes

    You must split array to small pieces.
    """
    assert type(vectors) is numpy.ndarray
    blosc.set_nthreads(nthreads)

    compressor = "lz4" if compress_type == COMPRESS_FASTEST else "zstd"
    level = 1 if compress_type == COMPRESS_FASTEST else 5
    buffer = blosc.compress_ptr(vectors.__array_interface__['data'][0], vectors.size,
                                typesize=max(1, min(255, vectors.dtype.itemsize)),
                                clevel=level, cname=compressor, shuffle=blosc.BITSHUFFLE)
    return pickle.dumps([buffer, vectors.dtype, vectors.shape])


def decompress_ndarray(binary: bytes, output_array=None) -> numpy.ndarray:
    """
    decompress_ndarray(binary[, output_array=None])

    Decompress array from buffer.
    Data will write in output_array if it is provided. Save time request memory spaces and avoid out of memory.
    """
    assert type(binary) is bytes
    assert type(output_array) in [numpy.ndarray, numpy.memmap, type(None)]

    buffer, dtype, shape = pickle.loads(binary)
    if output_array is None:
        output_array = numpy.empty(shape, dtype)
    else:
        assert dtype == output_array.dtype

        if type(output_array) is numpy.ndarray:
            output_array.resize(shape, refcheck=False)
        else:
            output_array = numpy.memmap(output_array.filename, shape=shape, dtype=dtype)

    assert output_array.shape == shape, "Array output's shape is't same as compressed array."

    blosc.decompress_ptr(buffer, output_array.__array_interface__['data'][0])
    return output_array


def compress_list(array: list, compress_type=COMPRESS_FASTEST, nthreads=blosc.ncores) -> bytes:
    """
    Handle from pickle and compress
    """
    assert type(array) is list
    return compress(pickle.dumps(array), compress_type=compress_type, nthreads=nthreads)


def decompress_list(buffer: bytes) -> list:
    """
    Handle from pickle and decompress
    """
    assert type(buffer) is bytes
    return pickle.loads(decompress(buffer))


__all__ = ['compress', 'decompress', 'compress_ndarray', 'decompress_ndarray', 'compress_list', 'decompress_list',
           'COMPRESS_BEST', 'COMPRESS_FASTEST']
