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


class VectorPool(object):
    __MODEL_EXT__ = "vecp"

    # TODO: buffer in ram that speed_up query.
    # TODO: multi-process to solve large Pool
    def __init__(self, dim, dtype=DEFAULT_TYPE):
        assert dtype in TYPE_SUPPORT
        assert dim > 0

        self.__file_path__ = tempfile.mkstemp()[1]
        self.__dim__ = dim
        self.__vector_pool__ = numpy.memmap(self.__file_path__, shape=(MIN_SIZE, self.__dim__), dtype=dtype)
        self.__table_writed__ = [False for _ in range(MIN_SIZE)]
        self.__table_ids__ = []
        self.__dtype__ = dtype
        self.__length__ = 0

    def __increase_pool_size(self, new_size):
        new_size = (int(new_size / ALPHA_SIZE) + 1) * 1024

        self.__table_writed__.extend([False for _ in range(self.__vector_pool__.shape[0], new_size)])
        self.__vector_pool__ = numpy.memmap(self.__file_path__, shape=(new_size, self.__dim__), dtype=self.__dtype__)

    def __del__(self):
        if os.path.abspath(self.__file_path__).split("/")[1] == "tmp":
            os.remove(self.__file_path__)

    def __eq__(self, other):
        return self.vectors() == other

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"Stored: {self.length} vectors, dim: {self.__dim__}, type: {self.dtype}\n" \
               f"{self.vectors().__repr__()}"

    def __getitem__(self, *args, **kwargs):
        return self.get(args)

    def __setitem__(self, ids, values):
        self.set(ids, values)

    def __iter__(self):
        """Loop generate"""
        return self.__vector_pool__.__iter__()

    def __add__(self, other):
        """equal a + b"""
        if isinstance(other, VectorPool):
            return self.vectors() + other.vectors
        return self.vectors() + other

    def __iadd__(self, other):
        """equal a += b"""
        if isinstance(other, VectorPool):
            self.__vector_pool__[self.__ids__] += other.vectors
        else:
            self.__vector_pool__[self.__ids__] += other
        return self

    def __sub__(self, other):
        """equal a - b"""
        if isinstance(other, VectorPool):
            return self.vectors() * other.vectors
        return self.vectors() - other

    def __isub__(self, other):
        """equal a -= b"""
        if isinstance(other, VectorPool):
            self.__vector_pool__[self.__ids__] -= other.vectors
        else:
            self.__vector_pool__[self.__ids__] -= other
        return self

    def __mul__(self, other):
        """equal a * b"""
        if isinstance(other, VectorPool):
            return self.vectors() * other.vectors
        return self.vectors() * other

    def __imul__(self, other):
        """equal a *= b"""
        if isinstance(other, VectorPool):
            self.__vector_pool__[self.__ids__] *= other.vectors
        else:
            self.__vector_pool__[self.__ids__] *= other
        return self

    def __truediv__(self, other):
        """equal a / b"""
        if isinstance(other, VectorPool):
            return self.vectors() / other.vectors
        return self.vectors() / other

    def __itruediv__(self, other):
        """equal a /= b"""
        if isinstance(other, VectorPool):
            self.__vector_pool__[self.__ids__] /= other.vectors
        else:
            self.__vector_pool__[self.__ids__] /= other
        return self

    def __pow__(self, power, modulo=None):
        return self.vectors()() ** power

    def __str__(self):
        return self.__repr__()

    def __abs__(self):
        """abs(a)"""
        return numpy.abs(self.vectors()())

    def __delitem__(self, key):
        self.remove(key)

    def __getattr__(self, item):
        assert hasattr(self.vectors(), item), "Not exist attribute!"
        return getattr(self.vectors(), item)

    def __exit__(self):
        self.__del__()

    @property
    def shape(self):
        return self.length, self.__dim__

    @property
    def vectors_size(self):
        return self.__dim__

    @property
    def length(self):
        return self.__length__

    def vectors(self):
        return self.__vector_pool__[self.__table_ids__, :]

    @property
    def dtype(self):
        return self.__vector_pool__.dtype

    def __check_input(self, vectors):
        assert isinstance(vectors, (tuple, list, numpy.ndarray, VectorPool))
        assert len(vectors.shape) == 2
        assert vectors.shape[1] == self.__dim__

    def add(self, vectors: numpy.ndarray) -> list:
        if isinstance(vectors, (list, tuple)):
            vectors = numpy.array(vectors, dtype=self.dtype)

        if len(vectors.shape) < 2 and vectors.shape[0] == self.__dim__:
            vectors = vectors.reshape(1, self.__dim__)

        if 'float' in str(vectors.dtype) and 'float' in str(self.dtype) and vectors.dtype != self.dtype:
            vectors = vectors.astype(self.dtype)

        self.__check_input(vectors)

        min_shape = self.length + vectors.shape[0]
        added_id = list(range(self.length, min_shape))

        while min_shape > self.__vector_pool__.shape[0]:
            self.__increase_pool_size(min_shape)

        write_ids = numpy.where(numpy.array(self.__table_writed__) == False)[0][:vectors.shape[0]]
        self.__table_ids__.extend(list(write_ids))

        self.__vector_pool__[write_ids, :] = vectors[:, :]
        for idx in write_ids:
            self.__table_writed__[idx] = True

        self.__length__ = min_shape
        return added_id

    def remove(self, ids):
        assert isinstance(ids, (int, list, numpy.ndarray))

        if isinstance(ids, numpy.ndarray):
            assert len(ids.shape) == 1 and 'int' in str(ids.dtype)

        if isinstance(ids, int):
            ids = [ids]

        remove_ids = numpy.array(self.__table_ids__)[ids]

        for idx in remove_ids:
            self.__table_writed__[idx] = False

        for idx in ids:
            del self.__table_ids__[idx]

        self.__length__ -= len(ids)
        return self.__vector_pool__[remove_ids, :]

    def pop(self, ids):
        return self.remove(ids)

    def clear(self):
        return self.remove(self.ids)

    def apply(self, func):
        assert hasattr(func, '__call__'), f"Require function. But got {type(func)}"
        self.__vector_pool__[self.__ids__] = func(self.vectors())

    def get(self, ids):
        assert isinstance(ids, (int, list, tuple, numpy.ndarray))
        get_ids = numpy.array(self.__table_ids__)[ids]
        return self.__vector_pool__[get_ids].view(type=numpy.ndarray)

    def set(self, ids, values):
        assert isinstance(ids, (int, list, tuple, slice, numpy.ndarray, VectorPool))

        if isinstance(values, VectorPool):
            values = values.vectors

        if isinstance(ids, int):
            ids = [ids]
            values = [values]

        if not isinstance(values, numpy.ndarray):
            values = numpy.array(values, dtype=self.dtype)

        self.__check_input(values)
        assert values.shape[0] == len(ids)

        set_ids = numpy.array(self.__table_ids__)[ids]
        self.__vector_pool__[set_ids] = values

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
        return self.__repr__()


__all__ = ['PERFORMANCE_TYPE', 'TYPE_SUPPORT', 'VectorPool']
