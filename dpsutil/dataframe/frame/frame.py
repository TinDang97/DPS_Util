import numpy

from dpsutil.compression import decompress, decompress_ndarray, compress, compress_ndarray
from dpsutil.media import imdecode, ENCODE_JPEG, DEFAULT_QUALITY, imencode


class Frame(object):
    """
    Frame Data

    work with vectors, image's frame and numpy.ndarray like.

    Support multi-type frame: buffer, numpy.ndarray
    Auto compress and decompress if dataframe is raw frame (pixel array - numpy.ndarray)
    """

    @property
    def size(self):
        if len(self.frame.shape) != 3:
            raise AttributeError
        return self.frame.shape[:2]

    @property
    def num_channels(self):
        if len(self.frame.shape) != 3:
            raise AttributeError
        return self.frame.shape[2]

    @classmethod
    def from_buffer(cls, data):
        header = data[0:1]
        data = data[1:]
        if header == b"\x00":
            return cls(decompress(data))

        if header == b"\x01":
            return cls(imdecode(data))
        return cls(decompress_ndarray(data))

    def tobytes(self):
        return self._tobytes()

    def _tobytes(self, compress_type=ENCODE_JPEG, quality=DEFAULT_QUALITY):
        if isinstance(self.frame, bytes):
            return b"\x00" + compress(self.frame)

        if self.frame.dtype == numpy.uint8:
            return b"\x01" + imencode(self.frame, compress_type, quality)
        return b"\x02" + compress_ndarray(self.frame)

    def encode(self, compress_type=ENCODE_JPEG, quality=DEFAULT_QUALITY):
        """
        Like `tobytes` function but require frame's dataframe is image bytearray.
        :return:
        """
        if not isinstance(self.frame, numpy.ndarray) or self.frame.dtype != numpy.uint8:
            raise TypeError("Only support image bytearray")
        return self._tobytes(compress_type, quality)

    @classmethod
    def decode(cls, data):
        """
        Like `from_buffer` function but only support if dataframe is image buffer.
        :param data:
        :return:
        """
        if data[:1] != b"\x01":
            raise TypeError("Decode only support image buffer")
        return cls.from_buffer(data)

    def __init__(self, frame, frame_size=None, dtype=None):
        if not isinstance(frame, (numpy.ndarray, bytes)):
            raise TypeError("Only support frame's type are `bytes` or `numpy.ndarray`")

        if frame_size:
            if not isinstance(frame_size, (tuple, list)):
                raise TypeError("Require frame_size is tuple or list")

            if len(frame_size) != 2:
                raise ValueError("Require frame_size is (width, height)!")

            if not isinstance(frame, numpy.ndarray):
                if dtype is None:
                    raise ValueError("Require dtype.")
                frame = numpy.frombuffer(frame, dtype=dtype)

            frame = frame.reshape((*frame_size, -1))

            if (num_channels := frame.shape[2]) not in [3, 4]:
                raise ValueError(f"Number channels of frame must be 3 (RGB, BGR) or 4 (ARGB, ABGR). Got {num_channels}")

        self.frame = frame

    def __repr__(self):
        if hasattr(self.frame, "shape"):
            return f"Frame\nShape: {self.frame.shape}\nRaw size: {self.frame.itemsize * self.frame.size} bytes."
        return f"Raw size: {str(self.frame.__len__())}"

    def __eq__(self, other):
        if not isinstance(other, types := (type(self), numpy.ndarray)):
            raise TypeError(f"Require: {types}")
        if isinstance(other, numpy.ndarray):
            return numpy.all(self.frame == other)
        return numpy.all(self.frame == other.frame)

    def __bytes__(self):
        return self.tobytes()
