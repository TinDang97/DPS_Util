import pickle
import blosc
import numpy

COMPRESS_FASTEST = 0
COMPRESS_BEST = 1

SPLIT_BYTES = b"<s<p>l>"


def compress(data: bytes, compress_type=COMPRESS_FASTEST, nthreads=blosc.ncores) -> bytes:
    assert type(data) is bytes
    blosc.set_nthreads(nthreads)

    compressor = "lz4" if compress_type == COMPRESS_FASTEST else "zstd"
    level = 1 if compress_type == COMPRESS_FASTEST else 5
    return blosc.compress(data, cname=compressor, clevel=level)


def decompress(binary: bytes) -> bytes:
    assert type(binary) is bytes
    return blosc.decompress(binary)


def compress_ndarray(vectors: numpy.ndarray, compress_type=COMPRESS_FASTEST, nthreads=blosc.ncores) -> bytes:
    assert type(vectors) is numpy.ndarray
    blosc.set_nthreads(nthreads)

    compressor = "lz4" if compress_type == COMPRESS_FASTEST else "zstd"
    level = 1 if compress_type == COMPRESS_FASTEST else 5
    buffer = blosc.compress_ptr(vectors.__array_interface__['data'][0], vectors.size,
                                typesize=max(1, min(255, vectors.dtype.itemsize)),
                                clevel=level, cname=compressor, shuffle=blosc.BITSHUFFLE)
    return pickle.dumps([buffer, vectors.dtype, vectors.shape])


def decompress_ndarray(binary: bytes, output_array=None) -> numpy.ndarray:
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
    assert type(array) is list
    return compress(pickle.dumps(array), compress_type=compress_type, nthreads=nthreads)


def decompress_list(buffer: bytes):
    assert type(buffer) is bytes
    return pickle.loads(decompress(buffer))


__all__ = ['compress', 'decompress', 'compress_ndarray', 'decompress_ndarray', 'compress_list', 'decompress_list',
           'COMPRESS_BEST', 'COMPRESS_FASTEST']
