import pickle
import warnings

import blosc
import numpy

from dpsutil.dataframe.convert import cvt_dec2hex, cvt_hex2dec, cvt_hex2str, cvt_str2hex

COMPRESS_FASTEST = 0
COMPRESS_BEST = 1

blosc.set_nthreads(min(8, max(4, blosc.detect_number_of_cores() // 2)))


def compress(data: bytes, compress_type=COMPRESS_FASTEST) -> bytes:
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

    compressor = "lz4" if compress_type == COMPRESS_FASTEST else "zstd"
    level = 1 if compress_type == COMPRESS_FASTEST else 5
    return blosc.compress(data, cname=compressor, clevel=level)


def decompress(data) -> bytes:
    """
    decompress(data: bytes)
    """
    return blosc.decompress(data)


def compress_ndarray(vectors, compress_type=COMPRESS_FASTEST) -> bytes:
    """
    compress_ndarray(vectors[, compress_type=COMPRESS_FASTEST, nthreads=blosc.ncores])

    High speed compress numpy.ndarray with multi-threading. Implement from blosc.compress

    Raises
    ------
    ValueError:
        if size of array larger than 2147483631 bytes.

        Example: array with float32 have itemsize=4 and size=614400000 ((1200000, 512) at 2D array)
        -> total size of array: 4*614400000 == 2457600000 bytes

    You must split array to small pieces.
    """
    if not isinstance(vectors, numpy.ndarray):
        raise TypeError("Only support numpy.ndarray type.")

    if vectors.dtype.itemsize == 1:
        warnings.warn(f"The compressor isn't effective with `{vectors.dtype}` type.")

    # prepare header data
    header = b""

    #   convert numpy dtype
    dtype_bytes = cvt_str2hex(str(vectors.dtype))
    header += cvt_dec2hex(len(dtype_bytes)) + dtype_bytes

    #   convert numpy shape
    for size_of_dim in vectors.shape:
        size_of_dim_bytes = cvt_dec2hex(size_of_dim)
        len_size_of_dim_bytes = 2 - len(size_of_dim_bytes)

        if len_size_of_dim_bytes > 0:
            size_of_dim_bytes = b"\x00" * len_size_of_dim_bytes + size_of_dim_bytes

        header += size_of_dim_bytes

    #   add info size of header
    header_size = len(header)
    if header_size > (1 << 8):
        raise ValueError("Vectors too large.")
    header = cvt_dec2hex(header_size) + header

    # compress vectors
    compressor = "lz4" if compress_type == COMPRESS_FASTEST else "zstd"
    level = 1 if compress_type == COMPRESS_FASTEST else 5
    buffer = blosc.compress_ptr(vectors.__array_interface__['data'][0], vectors.size,
                                typesize=max(1, min(255, vectors.dtype.itemsize)),
                                clevel=level, cname=compressor, shuffle=blosc.BITSHUFFLE)
    return header + buffer


def decompress_ndarray(binary, output_array=None) -> numpy.ndarray:
    """
    decompress_ndarray(binary[, output_array=None])

    Decompress array from buffer.
    Data will write in output_array if it is provided. Save time request memory spaces and avoid out of memory.

    Parameters
    ----------
    binary: bytes
        Numpy array bytes, which was compressed.

    output_array: numpy.ndarray
        Data after decompress will be write into output_array if give.
    """
    if not isinstance(binary, bytes):
        raise TypeError("Require byte type of input data.")

    if output_array is not None and not isinstance(output_array, (numpy.ndarray, numpy.memmap)):
        raise TypeError("Require numpy.ndarray type of output array.")

    cursor = 0

    # get header_size
    header_size = cvt_hex2dec(binary[cursor:cursor + 1])
    cursor += 1

    # get dtype_size
    dtype_size = cvt_hex2dec(binary[cursor: cursor + 1])
    cursor += 1

    # get dtype
    dtype = numpy.dtype(cvt_hex2str(binary[cursor: cursor + dtype_size]))
    cursor += dtype_size

    # get shape
    shape = []
    while 1:
        if cursor >= header_size + 1:
            break

        shape.append(cvt_hex2dec(binary[cursor: cursor + 2]))
        cursor += 2

    if output_array is None:
        output_array = numpy.empty(shape, dtype)
    else:
        if dtype != output_array.dtype:
            raise TypeError("Type of output array and data aren't the same!")

        if tuple(shape) != output_array.shape:
            if isinstance(output_array, numpy.memmap):
                output_array = numpy.memmap(output_array.filename, shape=shape, dtype=dtype)
            else:
                output_array.resize(shape, refcheck=False)

    blosc.decompress_ptr(binary[cursor:], output_array.__array_interface__['data'][0])
    return output_array


def compress_list(array: list, compress_type=COMPRESS_FASTEST) -> bytes:
    """
    Handle from pickle and compress
    """
    assert type(array) is list
    return compress(pickle.dumps(array), compress_type=compress_type)


def decompress_list(buffer: bytes) -> list:
    """
    Handle from pickle and decompress
    """
    assert type(buffer) is bytes
    return pickle.loads(decompress(buffer))


__all__ = ['compress', 'decompress', 'compress_ndarray', 'decompress_ndarray', 'compress_list', 'decompress_list',
           'COMPRESS_BEST', 'COMPRESS_FASTEST']
