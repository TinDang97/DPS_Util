from .attrdict import AttrDict, KeyNotFound
from ..compression import compress_list, decompress_list, COMPRESS_FASTEST


class EmptyKey(Exception):
    pass


class OutOfRange(Exception):
    pass


class TypeOfValueError(Exception):
    pass


class DefaultDict(AttrDict):
    """
    FixedDict help cover your dict with (keys, values) that was defined before. Implement from AttrDict

    Example:
        your_dict = FixedTypeDict(a=1, b=2)
        # {}
        # Default: {'a': 1, 'b': 2}
        # Definitely, your dict empty but that will be filled by default_value

        your_dict.a     # return: 1

        # Next, you set key a=5
        your_dict.a = 5

        your_dict.a     # return: 5

    User case:
    - When you need to control your params or hyper config.
    - Make sure your dict only store some field.
    """

    def __init__(self, **default_params):
        super().__init__({'default_params': AttrDict(default_params), "curr_params": AttrDict(default_params)})

    def __repr__(self):
        return self._current_params().__repr__()

    def __len__(self):
        return self._current_params().__len__()

    def keys(self):
        return self._current_params().keys()

    def items(self):
        return self._current_params().items()

    def values(self):
        return self._current_params().values()

    def default_params(self):
        return super().get('default_params')

    def _current_params(self):
        return super().get('curr_params')

    def __getitem__(self, item):
        if item in self._current_params():
            return self._current_params()[item]

    def __setitem__(self, key, value):
        if key not in self.default_params():
            raise KeyNotFound

        return self._current_params().__setitem__(key, value)

    def __delitem__(self, key):
        return self._current_params().__delitem__(key)

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    def __delattr__(self, item, force=False):
        try:
            return self.__delitem__(item)
        except KeyNotFound as e:
            if not force:
                raise e

    def __contains__(self, item):
        return self._current_params().__contains__(item)

    def __eq__(self, other):
        return self._current_params().__eq__(other)

    def __iter__(self):
        return self._current_params().__iter__()

    def __bool__(self):
        return self._current_params().__bool__()

    def __call__(self, data):
        assert isinstance(data, bytes)
        self.clear()
        self.from_buffer(data)
        return self

    def get(self, k):
        return self.__getitem__(k)

    def clear(self, *args):
        """
        Set value of key to default if provided.
        If not provide any key, clear all key of dict.
        :return:
        """
        if not args:
            self._current_params().update(self.default_params())
            return

        for k in args:
            self.__setitem__(k, self.default_params().__getitem__(k))

    def pop(self, k):
        """
        Like `clear` but require key.
        :return:
        """
        value = self[k]
        del self[k]
        return value

    def popitem(self):
        current_params = self._current_params().copy()
        self.clear()
        return current_params

    def from_array(self, arr):
        """
        Recover data from array.
        Make sure arrange of keys correctly.
        :param arr:
        :return:
        """
        assert isinstance(arr, list)
        if not self.__len__() == arr.__len__():
            raise OutOfRange(f"Require only {self.__len__()} params. But got {arr.__len__()}")

        for idx, key in enumerate(self):
            self[key] = arr[idx]

    def to_array(self):
        """
        Like `values` but return `list` instead.
        :return:
        """
        return list(self.values())

    def __bytes__(self):
        """
        This compress all values of dict. Ref: "to_array"
        :return:
        """
        return compress_list(self.to_array(), compress_type=COMPRESS_FASTEST)

    def from_buffer(self, buffer):
        """
        Decompress all values of dict. Ref: "from_array"

        ***Make sure arrange of keys correctly.
        :return:
        """
        assert isinstance(buffer, bytes)
        self.from_array(decompress_list(buffer))
        return self

    def to_buffer(self, compress_type=COMPRESS_FASTEST):
        return compress_list(self.to_array(), compress_type=compress_type)

    def setdefault(self, k, v=None, **kwargs):
        """
        Change default value of key.
        :return:
        """
        kwargs.update({k: v})
        self.default_params().update(kwargs)

    def update(self, params, **kwargs):
        """
        Fast way to set item via dict and kwargs
        :return:
        """
        assert isinstance(params, dict)

        params.update(kwargs)
        for k in params:
            if k in self.default_params():
                self[k] = params[k]


class DefaultTypeDict(DefaultDict):
    """
    FixedTypeDict help cover your dict when set item. Implement from FixedDict.

    Example:
        your_dict = FixedTypeDict(a=int, b=float)
        your_dict.a = 1     # It's working
        your_dict.a = 1.0   # Error TypeOfValueError
    """
    def __init__(self, **default_params):
        super().__init__()
        for k, v in default_params.items():
            if type(v) is not type:
                raise ValueError(f"Value must be class. {k}: {v}")
            self.setdefault(k, v)

    def __setitem__(self, key, value):
        if type(value) != self.default_params()[key]:
            raise TypeOfValueError
        super().__setitem__(key, value)

    def __getitem__(self, item):
        return self._current_params()[item]

    def type(self, key):
        return self.default_params()[key]


class UniqueTypeDict(DefaultDict):
    """
    Dict only access one type for all element.
    Raise TypeOfValueError if type of set value not same as type defined before.

    Example:
        your_dict = UniqueTypeDict(int)
        your_dict.a = 1     # it's working
        your_dict.a = 2.0   # raise error TypeOfValueError
    """
    def __init__(self, _type):
        super().__init__()
        if type(_type) is not type:
            raise TypeError(f"Only support type class. But got {_type}")
        self.setdefault('_type', _type)

    def __setitem__(self, key, value):
        if type(value) != self.type:
            raise TypeOfValueError

        self._current_params().__setitem__(key, value)

    @property
    def type(self):
        return self.default_params()['_type']


__all__ = ['DefaultDict', 'DefaultTypeDict', 'UniqueTypeDict', 'EmptyKey', 'OutOfRange', 'TypeOfValueError']
