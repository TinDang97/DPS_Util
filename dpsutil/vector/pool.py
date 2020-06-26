import os
import tempfile
from threading import Thread, Lock, Condition

import numpy

from dpsutil.compression import compress_ndarray, decompress_ndarray

AUTO_SIZE = -1
MIN_SIZE = 8

DEFAULT_TYPE = numpy.float32
PERFORMANCE_TYPE = numpy.float16
TYPE_SUPPORT = [numpy.float16, numpy.float32, numpy.float64, numpy.float128,
                numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
                numpy.int8, numpy.int16, numpy.int32, numpy.int64]

CACHE_DEFAULT = 256 * 1024 * 1024


class VectorPoolBase(object):
    """
    VectorPoolBase handle numpy.ndarray that save a lot of RAM by re-use existed memory space.
    Easier than numpy.ndarray. Likely list.
    - Reduce RAM
    - High Speed IO
    - Auto Scale
    - Support backup and recovery data in a second.

    *** Note:
    Cause thread safe, please call 'close' if not use another once.

    ** Benchmark with numpy.delete:

    ```
    timeit.timeit("numpy.delete(x, range(10, 20000), 0);" ,
    "import numpy; x = numpy.random.rand(1000000, 512).astype(numpy.float32);", number=1)
    ```
    timeit.timeit("a.remove(range(100, 20000))" , "import numpy; from dpsutil.vector.pool import VectorPool;
    a =  VectorPool(512, 100000); x = numpy.random.rand(1000000, 512).astype(numpy.float32); a.add(x)", number=1)
    ```

    - Faster than 2x
    - Using less memory than 1.5x
    """
    __MODEL_EXT__ = "vecp"

    def __init__(self, dim, size=MIN_SIZE, dtype=DEFAULT_TYPE):
        assert dim > 0
        assert dtype in TYPE_SUPPORT

        pool = self._init_pool((size, dim), dtype)

        assert len(pool.shape) == 2
        assert type(pool) in [numpy.ndarray, numpy.memmap]
        assert pool.shape[0] == size and size >= 0

        self._vector_pool = pool
        self.__dim = dim
        self.__used_indexes = []
        self.__unused_indexes = list(range(size))
        self.__dtype = dtype

        locker = Lock()
        self.__locker_clean = Condition(locker)
        self.__locker_write = Condition(locker)
        self.__stop_cleaner = False
        self.__relocating = False
        self.__interrupt_processing = False
        Thread(target=self.__relocation, daemon=True).start()

    def __relocation(self):
        while 1:
            with self.__locker_clean:
                self.__locker_clean.wait()
                if self.__stop_cleaner:
                    break

                if not self.__unused_indexes or not self.__used_indexes:
                    continue

                self.__relocating = True
                get_indexes = numpy.argsort(-1 * numpy.asarray(self.__used_indexes))
                write_indexes = numpy.argsort(numpy.asarray(self.__unused_indexes))

                for get_index, write_index in zip(get_indexes, write_indexes):
                    if self.__interrupt_processing:
                        break

                    get_idx = self.__used_indexes[get_index]
                    write_idx = self.__unused_indexes[write_index]

                    if get_idx <= write_idx:
                        break

                    self._vector_pool[write_idx] = self._vector_pool[get_idx]
                    self.__unused_indexes[write_index] = get_idx
                    self.__used_indexes[get_index] = write_idx

                if self.__interrupt_processing:
                    self.__interrupt_processing = False
                else:
                    self._decrease_pool_size(self.__unused_indexes.__len__())
                    self.__unused_indexes.clear()

                self.__relocating = False
                self.__locker_write.notify()

    def _init_pool(self, shape, dtype):
        raise NotImplementedError("Not initial pool.")

    def _increase_pool_size(self, size):
        raise NotImplementedError("Pool must handle this method to increase pool size.")

    def _decrease_pool_size(self, size):
        raise NotImplementedError("Pool must handle this method to decrease pool size.")

    def __len__(self):
        return self.__used_indexes.__len__()

    def __repr__(self):
        return f"Stored: {self.__len__()} vectors, dim: {self.__dim}, type: {self.dtype}\n"

    def __getitem__(self, *args, **kwargs):
        return self.get(args)

    def __setitem__(self, indexes, values):
        self.set(indexes, values)

    def __iter__(self):
        """Loop generate"""
        return self._vector_pool.__iter__()

    def __str__(self):
        return self.__repr__()

    def __delitem__(self, key):
        self.remove(key)

    def __eq__(self, other):
        raise AttributeError("Comparator was not supported. Instead, call 'vectors' method before compare.")

    def __lt__(self, other):
        self.__eq__(other)

    def __le__(self, other):
        self.__eq__(other)

    def __ne__(self, other):
        self.__eq__(other)

    def __gt__(self, other):
        self.__eq__(other)

    def __ge__(self, other):
        self.__eq__(other)

    @property
    def shape(self) -> tuple:
        """
        :return: (length, dim)
        """
        return self.__len__(), self.__dim

    @property
    def vectors_size(self) -> int:
        """
        :return: dim of vector
        """
        return self.__dim

    def vectors(self) -> numpy.ndarray:
        """
        return all vectors in pool
        """
        with self.__locker_write:
            if self.__relocating:
                self.__interrupt_processing = True
                self.__locker_write.wait()

            output = self._vector_pool[self.__used_indexes]
            self.__locker_clean.notify()
        return output

    @property
    def dtype(self):
        """
        Type of pool
        """
        return self.__dtype

    @property
    def dim(self):
        return self.__dim

    def _norm_input(self, indexes, vectors):
        assert isinstance(vectors, numpy.ndarray)
        assert type(indexes) in [int, list, slice, tuple, range, numpy.ndarray]

        if type(indexes) is int and vectors.ndim == 1 and vectors.size == self.__dim:
            vectors = vectors.reshape(1, self.__dim)

        if 'float' in str(vectors.dtype) and 'float' in str(self.dtype) and vectors.dtype != self.dtype:
            vectors = vectors.astype(self.dtype)

        assert vectors.shape[1] == self.__dim
        assert vectors.dtype == self.dtype

        if type(indexes) is int:
            indexes = [indexes]

        if type(indexes) is tuple and len(indexes) == 1:
            indexes = indexes[0]

        if type(indexes) is slice:
            indexes = range(indexes.start if indexes.start else 0,
                            indexes.stop if indexes.stop else self._vector_pool.shape[0],
                            indexes.step if indexes.step else 1)

        if type(indexes) is numpy.ndarray:
            assert "int" in str(indexes.dtype)

        return indexes, vectors

    def add(self, vectors: numpy.ndarray):
        assert isinstance(vectors, numpy.ndarray)

        if vectors.ndim == 1 and vectors.size == self.__dim:
            vectors = vectors.reshape(1, self.__dim)

        if 'float' in str(vectors.dtype) and 'float' in str(self.dtype) and vectors.dtype != self.dtype:
            vectors = vectors.astype(self.dtype)

        # increase pool size that have enough space to write new data
        with self.__locker_write:
            if self.__relocating:
                self.__interrupt_processing = True
                self.__locker_write.wait()

            if self.__unused_indexes.__len__() < vectors.shape[0]:
                current_size = self._vector_pool.shape[0]
                increase_size = vectors.shape[0] - self.__unused_indexes.__len__()
                self.__unused_indexes.extend(range(current_size, current_size + increase_size))
                self._increase_pool_size(increase_size)

            write_indexes = range(vectors.shape[0])
            empty_lst = numpy.asarray(self.__unused_indexes)

            mask = numpy.ones(empty_lst.size, dtype=numpy.bool)
            mask[write_indexes] = False

            write_indexes = empty_lst[write_indexes]
            self._vector_pool[write_indexes] = vectors
            self.__used_indexes.extend(write_indexes)

            self.__unused_indexes = empty_lst[mask].tolist()
            self.__locker_clean.notify()

    def insert(self, indexes, vectors):
        indexes, vectors = self._norm_input(indexes, vectors)

        with self.__locker_write:
            if self.__relocating:
                self.__interrupt_processing = True
                self.__locker_write.wait()

            if self.__unused_indexes.__len__() < indexes.__len__():
                current_size = self._vector_pool.shape[0]
                increase_size = vectors.shape[0] - self.__unused_indexes.__len__()
                self.__unused_indexes.extend(range(current_size, current_size + increase_size))
                self._increase_pool_size(increase_size)

            write_indexes = [self.__unused_indexes.pop() for _ in range(len(indexes))]
            self._vector_pool[write_indexes] = vectors

            for idx, write_index in zip(indexes, write_indexes):
                self.__used_indexes.insert(idx, write_index)
            self.__locker_clean.notify()

    def remove(self, indexes):
        assert isinstance(indexes, (int, list, numpy.ndarray, range))

        if isinstance(indexes, numpy.ndarray):
            assert len(indexes.shape) == 1 and 'int' in str(indexes.dtype)

        if isinstance(indexes, int):
            indexes = [indexes]

        if numpy.max(indexes) >= self.__used_indexes.__len__():
            assert IndexError("Out of range!")

        with self.__locker_write:
            if self.__relocating:
                self.__interrupt_processing = True
                self.__locker_write.wait()

            # marked position to unwritten
            lst = numpy.asarray(self.__used_indexes)
            remove_lst = lst[indexes]

            mask = numpy.ones(lst.size, dtype=numpy.bool)
            mask[indexes] = False
            list_remaining = lst[mask]

            self.__unused_indexes.extend(remove_lst.tolist())
            self.__used_indexes = list(list_remaining)
            self.__locker_clean.notify()

    def pop(self, idx=0):
        assert type(idx) is int

        with self.__locker_write:
            if self.__relocating:
                self.__interrupt_processing = True
                self.__locker_write.wait()

            remove_idx = self.__used_indexes.pop(idx)
            self.__unused_indexes.append(remove_idx)
            output = self._vector_pool[remove_idx].copy()
            self.__locker_clean.notify()
        return output

    def clear(self):
        # delete marked point
        with self.__locker_write:
            if self.__relocating:
                self.__interrupt_processing = True
                self.__locker_write.wait()

            self.__unused_indexes.clear()
            self.__used_indexes.clear()
            self._decrease_pool_size(self._vector_pool.shape[0])
            self.__locker_clean.notify()

    def close(self):
        with self.__locker_write:
            if self.__relocating:
                self.__interrupt_processing = True
                self.__locker_write.wait()

            self.__unused_indexes.clear()
            self.__used_indexes.clear()
            self._decrease_pool_size(self._vector_pool.shape[0])
            self.__stop_cleaner = True
            self.__locker_clean.notify()

    def get(self, indexes):
        assert isinstance(indexes, (int, list, tuple, numpy.ndarray))

        if type(indexes) is tuple and len(indexes) == 1:
            indexes = indexes[0]

        with self.__locker_write:
            if self.__relocating:
                self.__interrupt_processing = True
                self.__locker_write.wait()

            if type(indexes) in [int, slice]:
                get_indexes = self.__used_indexes.__getitem__(indexes)
            else:
                get_indexes = [self.__used_indexes[idx] for idx in indexes]

            output = self._vector_pool[get_indexes]
            self.__locker_clean.notify()
        return output

    def set(self, indexes, vectors):
        indexes, vectors = self._norm_input(indexes, vectors)

        if numpy.max(indexes) >= self.__used_indexes.__len__():
            assert IndexError("Out of range!")

        with self.__locker_write:
            if self.__relocating:
                self.__interrupt_processing = True
                self.__locker_write.wait()

            write_indexes = [self.__used_indexes[idx] for idx in indexes]
            self._vector_pool[write_indexes] = vectors
            self.__locker_clean.notify()

    def save(self, file_name, folder=".", over_write=False, indexes=None) -> str:
        """
        Save all compressed array into file.

        Raise ValueError if size of array larger than 2147483631 bytes.

        Example: array with float32 have itemsize=4 and size=614400000 ((1200000, 512) at 2D array)
        -> total size of array: 4*614400000 == 2457600000 bytes

        You must split array to small pieces.

        * File's path is defined by `{folder's path}/{file_name + POOL.__MODEL_EXT__}`
        """
        assert isinstance(file_name, str)
        assert self.shape[0] != 0, "Vector pool is empty!"

        file_path = f"{folder}/{file_name}.{self.__MODEL_EXT__.lower()}"
        file_path = os.path.abspath(file_path)
        assert not os.path.isfile(file_path) or over_write, f"{str(FileExistsError.__name__)}" \
                                                            "\nIf you want to overwrite: over_write=True"

        with self.__locker_write:
            if self.__relocating:
                self.__interrupt_processing = True
                self.__locker_write.wait()

            with open(file_path, 'wb') as f:
                if indexes is None:
                    vectors = self._vector_pool[self.__used_indexes]
                else:
                    get_indexes = self.__used_indexes[indexes]
                    vectors = self._vector_pool[get_indexes]
                f.write(compress_ndarray(vectors))
            self.__locker_clean.notify()
        return file_path

    def load(self, file_path):
        """
        Load existed vectors from buffer.
        Note: Data will be cleaned after loaded.
        :return:
        """
        assert os.path.isfile(file_path), str(FileNotFoundError.__name__)
        assert file_path.split(".")[-1].lower() == self.__MODEL_EXT__, \
            f"Only support .{self.__MODEL_EXT__} or {self.__MODEL_EXT__.lower()} file!"

        with self.__locker_write:
            if self.__relocating:
                self.__interrupt_processing = True
                self.__locker_write.wait()

            with open(file_path, 'rb') as f:
                self._vector_pool = decompress_ndarray(f.read(), output_array=self._vector_pool)

            self.__unused_indexes.clear()
            self.__used_indexes = list(range(self._vector_pool.__len__()))
            self.__locker_clean.notify()

    def from_file(self, file_path):
        """
        Add vectors from file
        """
        assert os.path.isfile(file_path), str(FileNotFoundError.__name__)
        assert file_path.split(".")[-1].lower() == self.__MODEL_EXT__, \
            f"Only support .{self.__MODEL_EXT__} or {self.__MODEL_EXT__.lower()} file!"

        with self.__locker_write:
            if self.__relocating:
                self.__interrupt_processing = True
                self.__locker_write.wait()

            with open(file_path, 'rb') as f:
                self.from_buffer(f.read())

    def from_buffer(self, buffer):
        """
        Add vectors from buffer
        """
        assert type(buffer) is bytes

        with self.__locker_write:
            if self.__relocating:
                self.__interrupt_processing = True
                self.__locker_write.wait()

            vectors = decompress_ndarray(buffer)
            self.add(vectors)
            self.__locker_clean.notify()

    def to_bytes(self, indexes=None) -> bytes:
        """
        Compress array

        Raise ValueError if size of array larger than 2147483631 bytes.

        Example: array with float32 have itemsize=4 and size=614400000 ((1200000, 512) at 2D array)
        -> total size of array: 4*614400000 == 2457600000 bytes

        You must split array to small pieces.
        """
        with self.__locker_write:
            if self.__relocating:
                self.__interrupt_processing = True
                self.__locker_write.wait()

            if indexes is None:
                vectors = self._vector_pool[self.__used_indexes]
            else:
                get_indexes = self.__used_indexes[indexes]
                vectors = self._vector_pool[get_indexes]

            output = compress_ndarray(vectors)
            self.__locker_clean.notify()
        return output


class VectorPool(VectorPoolBase):
    """
    Vectors still store in RAM. Perfect for small pool.
    IO faster than VectorPoolMMap about 1.5x.
    If your ram isn't enough, you can use VectorPoolMMap better for large pool, surely it take a little time.

    *** Note:
    Cause thread safe, please call 'close' if not use another once.
    """

    def _init_pool(self, shape, dtype):
        return numpy.zeros(shape, dtype=dtype)

    def _increase_pool_size(self, size):
        self._vector_pool.resize(self._vector_pool.shape[0] + size, self.dim)

    def _decrease_pool_size(self, size):
        self._vector_pool.resize(max(self._vector_pool.shape[0] - size, MIN_SIZE), self.dim)


class VectorPoolMMap(VectorPoolBase):
    """
    Vectors still store in Disk. Perfect for large pool.
    If your ram is enough for large pool, you can use VectorPool with High Speed IO.

    This implement from numpy.memmap. It already handle cache buffer in os kernel level.
    If you want to data in RAM. Call 'copy' method after 'get'.

    *** Note:
    Cause thread safe, please call 'close' if not use another once.
    """

    def _init_pool(self, shape, dtype):
        return numpy.memmap(tempfile.mkstemp()[1], shape=shape, dtype=dtype)

    def _increase_pool_size(self, size):
        self._vector_pool = numpy.memmap(self._vector_pool.filename,
                                         shape=(self._vector_pool.shape[0] + size, self.dim), dtype=self.dtype)

    def _decrease_pool_size(self, size):
        self._vector_pool = numpy.memmap(self._vector_pool.filename,
                                         shape=(max(self._vector_pool.shape[0] - size, MIN_SIZE), self.dim,),
                                         dtype=self.dtype)


__all__ = ['PERFORMANCE_TYPE', 'DEFAULT_TYPE', 'TYPE_SUPPORT', 'VectorPool', 'VectorPoolMMap']
