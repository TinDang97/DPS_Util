import attrdict

from dpsutil.compression import compress_list, decompress_list, COMPRESS_FASTEST
from dpsutil.hash import short_hash


class AttrDict(attrdict.AttrDict):
    """
    AttrDict will help you get value of key via attributes. Implement attrdict.AttrDict
    More info here: https://github.com/bcj/AttrDict

    AttrDict() -> new empty dictionary
    AttrDict(mapping) -> clone a dict
    AttrDict(**kwargs) -> new dictionary initialized with the name=value pairs
    AttrDict(Iterable[key, value]) -> create dict with pair k=v
    AttrDict(Iterable[key][, value=None]: Optional) -> create dict with default value, None if it's not gave.

    Example:
        _dict = {'a': 1, 'b': 1)

        # initial
        your_dict = AttrDict(_dict)
        your_dict = AttrDict(**_dict) equal AttrDict(a=1, b=1)
        your_dict = AttrDict({'a', 'b'}, 1)
        your_dict = AttrDict(zip(['a', 'b'], [1, 1]))

        your_dict.a     # return: 1
        your_dict.b     # return: 1
        your_dict.c     # raise KeyError

    ==================
    Supported decorator.
        @attrdict.attribute_dict

    Decorator that it create dict by attribute of class.
    Support attribute alias:
        @attrdict.attribute_dict
        class CustomDict:
            a=1
            b=2

        custom_dict = CustomDict()
        custom_dict.a   # return: 1
        custom_dict.b   # return: 2
    """

    def __init__(self, __dict=None, __value=None, **default_params):
        # create instance
        super().__init__()
        self._setattr('_sequence_type', list)

        # update data from constructor
        if __dict is None:
            __dict = {}

        if isinstance(__dict, str):
            __dict = [__dict]

        if isinstance(__dict, dict):
            __dict = __dict.items()

        for _dict in [__dict, default_params.items()]:
            for pair in _dict:
                if type(pair) is tuple:
                    k, v = pair
                else:
                    k = pair
                    v = __value
                self[k] = v

    def __repr__(self):
        string = ""
        for k, v in self.items():
            if type(v) is str:
                v = f"'{v}'"
            if string:
                string = f"{string} | {k}={v}"
            else:
                string = f"{k}={v}"
        return f"{{{string}}}"

    def copy(self):
        return AttrDict(self)

    def clear(self, *args):
        if not args:
            return super().clear()

        for k in args:
            self.__delitem__(k)

    def _setattr(self, key, value):
        if key.startswith("__"):
            key = f"_{self.__class__.__name__}{key}"
        super(AttrDict, self)._setattr(key, value)

    def from_buffer(self, buffer):
        """
        Decompress all values of dict. Ref: "from_array"

        ***Make sure arrange of keys correctly.
        :return:
        """
        assert isinstance(buffer, bytes)

        # decompress
        _data = decompress_list(buffer)
        _hash_headers = _data.pop(0)

        # check hash of headers
        if _hash_headers != short_hash(''.join(self.keys())):
            raise BufferError("The keys of current dict aren't correct.")

        for idx, key in enumerate(self.keys()):
            self[key] = _data[idx]

    def to_buffer(self, compress_type=COMPRESS_FASTEST) -> bytes:
        """
        Compress this dict to bytes.

        compress_type: COMPRESS_FASTEST|COMPRESS_BEST
        """
        _data = []

        # prepare data
        _hash_headers = short_hash(''.join(self.keys()))
        _values = self.values()

        # add data
        _data.append(_hash_headers)
        _data.extend(_values)

        # compress
        return compress_list(_data, compress_type=compress_type)

    def __bytes__(self) -> bytes:
        """
        This compress all values of dict. Ref: "to_array"
        :return:
        """
        return compress_list(self.to_array(), compress_type=COMPRESS_FASTEST)

    @classmethod
    def fromkeys(cls, *args, **kwargs):
        return cls(*args, **kwargs)


__all__ = ['AttrDict']
