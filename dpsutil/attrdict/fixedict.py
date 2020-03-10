from .attrdict import AttrDict, KeyNotFound
from ..compression import compress_list, decompress_list


class EmptyKey(Exception):
    pass


class OutOfRange(Exception):
    pass


class TypeOfValueError(Exception):
    pass


class FixedDict(AttrDict):
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
    - When you need to control your params. Like config of AI Model.
    - Make sure your dict only store some field.
    """

    def __init__(self, **default_params):
        if not default_params:
            raise EmptyKey("Not existed any keys.")

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

    def __iter__(self):
        return iter(self._current_params())

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

    def get(self, k):
        return self.__getitem__(k)

    def clear(self, *args):
        if not args:
            self.from_dict(self.default_params())
            return

        for k in args:
            del self[k]

    def pop(self, k):
        value = self[k]
        del self[k]
        return value

    def from_array(self, arr):
        assert isinstance(arr, list)
        if not self.__len__() == arr.__len__():
            raise OutOfRange(f"Require only {self.__len__()} params. But got {arr.__len__()}")

        for idx, key in enumerate(self):
            self[key] = arr[idx]

    def to_array(self):
        return list(self.values())

    def __bytes__(self):
        return compress_list(self.to_array())

    def from_buffer(self, buffer):
        assert isinstance(buffer, bytes)
        self.from_array(decompress_list(buffer))
        return self

    def setdefault(self, k, d=None):
        if k not in self.default_params():
            raise KeyNotFound
        self.default_params().__setitem__(k, d)

    def update(self, params, **kwargs):
        assert isinstance(params, dict)

        params.update(kwargs)
        for k in params:
            if k in self.default_params():
                self[k] = params[k]


class FixedTypeDict(FixedDict):
    """
    FixedTypeDict help cover your dict when set item. Implement from FixedDict.

    Example:
        your_dict = FixedTypeDict(a=int, b=float)
        your_dict.a = 1     # It's working
        your_dict.a = 1.0   # Error TypeOfValueError
    """
    def __init__(self, **default_params):
        super().__init__(**default_params)
        for k in default_params:
            del self[k]

    def __setitem__(self, key, value):
        if type(value) != self.default_params()[key]:
            raise TypeOfValueError
        super().__setitem__(key, value)

    def __getitem__(self, item):
        return self._current_params()[item]

    def type(self, key):
        return self.default_params()[key]


__all__ = ['FixedDict', 'FixedTypeDict', 'EmptyKey', 'OutOfRange', 'TypeOfValueError']
