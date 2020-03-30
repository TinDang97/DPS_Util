import os
import numpy
import tempfile

from ..compression import compress_ndarray, decompress_ndarray

AUTO_SIZE = -1
MIN_SIZE = 256
ALPHA_SIZE = 1024

DEFAULT_TYPE = numpy.float32
PERFORMANCE_TYPE = numpy.float16
TYPE_SUPPORT = [numpy.float16, numpy.float32, numpy.float64, numpy.float128,
                numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
                numpy.int8, numpy.int16, numpy.int32, numpy.int64]


class VectorPoolBase(object):
    """
    VectorPool implement numpy.ndarray to save all vectors in disk with High Performance IO.
    Easier than numpy.memmap. Likely list.
    - Reduce RAM
    - Speedup IO
    - Auto Scale
    - Support backup and recovery data in a second.
    """
    __MODEL_EXT__ = "vecp"

    def __init__(self, pool, size=MIN_SIZE, dtype=DEFAULT_TYPE):
        assert dtype in TYPE_SUPPORT
        assert len(pool.shape) == 2
        assert type(pool) in [numpy.ndarray, numpy.memmap]
        assert pool.shape[0] == size and size > 0

        self._vector_pool = pool
        self._table_wrote = numpy.full(size, False, dtype=numpy.bool)
        self._table_ids = []

        self.__dtype__ = dtype
        self.__length__ = 0
        self._dim = pool.shape[1]

    def _increase_pool_size(self, new_size):
        new_size = (int(new_size / ALPHA_SIZE) + 1) * 1024
        
        self._table_wrote = list(self._table_wrote)

        for _ in range(self._vector_pool.shape[0], new_size):
            self._table_wrote.append(False)
        
        self._table_wrote = numpy.array(self._table_wrote, dtype=numpy.bool)
        return new_size

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"Stored: {self.length} vectors, dim: {self._dim}, type: {self.dtype}\n" \
               f"{self.vectors().__repr__()}"

    def __getitem__(self, *args, **kwargs):
        return self.get(args)

    def __setitem__(self, ids, values):
        self.set(ids, values)

    def __iter__(self):
        """Loop generate"""
        return self._vector_pool.__iter__()

    def __str__(self):
        return self.__repr__()

    def __delitem__(self, key):
        self.remove(key)

    @property
    def shape(self):
        return self.length, self._dim

    @property
    def vectors_size(self):
        return self._dim

    @property
    def length(self):
        return self.__length__

    def vectors(self):
        return self._vector_pool[self._table_ids]

    @property
    def dtype(self):
        return self._vector_pool.dtype

    def add(self, vectors: numpy.ndarray):
        assert isinstance(vectors, numpy.ndarray)

        if len(vectors.shape) < 2 and vectors.shape[0] == self._dim:
            vectors = vectors.reshape(1, self._dim)

        if 'float' in str(vectors.dtype) and 'float' in str(self.dtype) and vectors.dtype != self.dtype:
            vectors = vectors.astype(self.dtype)

        min_shape = self.length + vectors.shape[0]
        self._increase_pool_size(min_shape)

        write_ids = numpy.where(self._table_wrote == False)[0][:vectors.shape[0]]

        for idx in write_ids:
            self._table_ids.append(idx)
            self._table_wrote[idx] = True

        self._vector_pool[write_ids, :] = vectors
        self.__length__ = min_shape

    def remove(self, ids):
        assert isinstance(ids, (int, list, numpy.ndarray))

        if isinstance(ids, numpy.ndarray):
            assert len(ids.shape) == 1 and 'int' in str(ids.dtype)

        if isinstance(ids, int):
            ids = [ids]

        remove_ids = [self._table_ids[idx] for idx in ids]

        # delete marked point
        for idx in remove_ids:
            self._table_wrote[idx] = False

        for idx in ids:
            del self._table_ids[idx]

        self.__length__ -= len(ids)

    def pop(self, idx):
        assert type(idx) is int

        remove_idx = self._table_ids[idx]
        self._table_wrote[remove_idx] = False
        del self._table_ids[idx]
        self.__length__ -= 1
        return self._vector_pool[remove_idx]

    def clear(self):
        # delete marked point
        self._table_wrote[self._table_ids] = False
        self._table_ids = []
        self.__length__ = 0

    def __get_id(self, ids):
        assert isinstance(ids, (int, list, tuple, numpy.ndarray))

        if type(ids) is int:
            get_ids = self._table_ids[ids]
        elif type(ids) is tuple:
            assert len(ids) == 1 and type(ids[0]) == slice
            get_ids = self._table_ids.__getitem__(ids[0])
        else:
            get_ids = [self._table_ids[idx] for idx in ids]
        return get_ids

    def get(self, ids):
        return self._vector_pool[self.__get_id(ids)]

    def set(self, ids, vectors):
        assert isinstance(vectors, numpy.ndarray)

        if isinstance(ids, int):
            assert len(vectors.shape) == 1 and vectors.shape[0] == self._dim
            ids = [ids]
            vectors = vectors.reshape(1, self._dim)

        assert vectors.shape[0] == len(ids)
        assert vectors.shape[1] == self._dim

        self._vector_pool[self.__get_id(ids)] = vectors

    def save(self, file_name, folder=".", over_write=False):
        assert isinstance(file_name, str)
        assert self.shape[0] != 0, "Vector pool is empty!"

        file_path = f"{folder}/{file_name}.{self.__MODEL_EXT__.lower()}"
        file_path = os.path.abspath(file_path)
        assert not os.path.isfile(file_path) or over_write, f"{str(FileExistsError.__name__)}" \
                                                            "\nIf you want to overwrite: over_write=True"

        with open(file_path, 'wb') as f:
            f.write(compress_ndarray(self.vectors()))
            f.flush()

        return file_path

    def load(self, file_path):
        assert os.path.isfile(file_path), str(FileNotFoundError.__name__)
        assert file_path.split(".")[-1].lower() == self.__MODEL_EXT__, \
            f"Only support .{self.__MODEL_EXT__} or {self.__MODEL_EXT__.lower()} file!"

        with open(file_path, 'rb') as f:
            vector_pool = decompress_ndarray(f.read())
        self.add(vector_pool)


class VectorPool(VectorPoolBase):
    def __init__(self, dim, size=MIN_SIZE, dtype=DEFAULT_TYPE):
        pool = numpy.empty((size, dim), dtype=dtype)
        super().__init__(pool, size=size, dtype=dtype)

    def _increase_pool_size(self, new_size):
        new_size = super()._increase_pool_size(new_size)
        self._vector_pool.resize(new_size, self._dim)


class VectorPoolMMap(VectorPoolBase):
    def __init__(self, dim, size=MIN_SIZE, dtype=DEFAULT_TYPE):
        self._file_path = tempfile.mkstemp()[1]
        pool = numpy.memmap(self._file_path, shape=(size, dim), dtype=dtype)
        super().__init__(pool, size=size, dtype=dtype)

    def _increase_pool_size(self, new_size):
        new_size = super()._increase_pool_size(new_size)
        self._vector_pool = numpy.memmap(self._file_path, shape=(new_size, self._dim), dtype=self.__dtype__)

    def __del__(self):
        if os.path.abspath(self._file_path).split("/")[1] == "tmp":
            os.remove(self._file_path)

    def __exit__(self):
        self.__del__()


class VectorPoolCached(VectorPoolBase):
    pass


__all__ = ['PERFORMANCE_TYPE', 'DEFAULT_TYPE', 'TYPE_SUPPORT', 'VectorPool', 'VectorPoolMMap']
