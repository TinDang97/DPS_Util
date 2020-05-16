from inspect import isclass
from typing import Tuple, Any, List

from .attrdict import AttrDict


class DefaultDict(AttrDict):
    """
    DefaultDict help cover your dict with (keys, values) that was defined before.
    Implement from dpsutil.attrdict.AttrDict

    Example:
        your_dict = DefaultDict(a=1, b=2)
        your_dict.a    # return: 1

        # Set key-value if it wasn't defined before
        your_dict.c = 4   # raise KeyError

        # To avoid KeyError
        your_dict.set_default('c', 4)

        # Next, you set key a=5
        your_dict.a = 5
        your_dict.a    # return: 5

        # After clear value, default value will be set.
        your_dict.clear('a')
        or
        del your_dict['a']

        your_dict.a    # return: 1

        # Delete key.
        your.clear('a')
        /// or 
        del your_dict['a']

        # Remove key:
        your_dict.del_default('a')
        or
        your_dict.remove('a')

    ==================
    Supported decorator.
        @attrdict.default_dict

    Decorator that it create DefaultDict by attribute of class.
    Support attribute alias:
        @attrdict.default_dict
        class CustomDict:
            a=1
            b=2

        custom_dict = CustomDict()
        custom_dict.a   # return: 1
        custom_dict.b   # return: 2

    User case:
    - When you need to control your params or hyper config.
    - Make sure your dict only store some field.
    """

    def _default_contain(self, key):
        return key in super().__getattribute__(f"_{self.__class__.__name__}__default")

    def get_default(self, key):
        return super().__getattribute__(f"_{self.__class__.__name__}__default")[key]

    def del_default(self, key):
        return self.pop(key)

    def _del_default(self, key) -> Tuple[Any]:
        return super().__getattribute__(f"_{self.__class__.__name__}__default").__delitem__(key)

    def setdefault(self, _k=None, _v=None, **kwargs):
        """
        Change default value of key.

        Example:
        a = {k: v}

        for k, v in a.items():
            defaultdict.setdefault(k, v)

        or

        defaultdict.setdefault(**a)
        """
        configs_default = AttrDict(_k, _v, **kwargs)
        super().__getattribute__(f"_{self.__class__.__name__}__default").update(configs_default)

    def __init__(self, *args, **default_params):
        self._setattr("__default", {})
        self.setdefault(**AttrDict(*args, **default_params))
        super().__init__()

    def __iter__(self):
        return iter(self.keys())

    def __setitem__(self, key, value):
        if key not in super().__getattribute__(f"_{self.__class__.__name__}__default"):
            raise KeyError("Key not in default keys.")
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        try:
            value = super().__getitem__(key)
        except KeyError as e:
            if key not in super().__getattribute__(f"_{self.__class__.__name__}__default"):
                raise e
            value = self.get_default(key)
        return value

    def __delitem__(self, key):
        """
        Clear and set back value to default.
        """
        try:
            super().__delitem__(key)
        except KeyError as e:
            if self.get_default(key) is None:
                raise e

    def __setattr__(self, key, value):
        if key not in super().__getattribute__(f"_{self.__class__.__name__}__default"):
            raise KeyError("Key not in default keys.")
        return super().__setattr__(key, value)

    def __getattr__(self, key):
        try:
            value = super().__getattr__(key)
        except AttributeError as e:
            if key not in super().__getattribute__(f"_{self.__class__.__name__}__default"):
                raise e
            value = self.get_default(key)
        return value

    def __delattr__(self, key, **kwargs):
        """
        Clear and set back value to default.
        """
        try:
            super().__delattr__(key)
        except KeyError as e:
            if key not in super().__getattribute__(f"_{self.__class__.__name__}__default"):
                raise e

    def __call__(self, _data):
        if type(_data) is bytes:
            self.from_buffer(_data)
        elif isinstance(_data, dict):
            self.update(_data)
        else:
            raise TypeError(f"Type {type(_data)} isn't supported!")
        return self

    def __len__(self) -> int:
        return self.__getattribute__(f"_{self.__class__.__name__}__default").__len__()

    def __eq__(self, other):
        if not isinstance(other, dict):
            raise TypeError(f"Can't compare with {other.__class__.__name__}")
        return dict(self).__eq__(dict(other))

    def get(self, key):
        try:
            value = self.__getitem__(key)
        except KeyError:
            return None
        return value

    def values(self) -> List[Any]:
        """
        Return all values of dict
        """
        for k in self.keys():
            yield self.__getitem__(k)

    def keys(self) -> List[str]:
        """
        Return all keys of dict
        """
        return self.__getattribute__(f"_{self.__class__.__name__}__default").keys()

    def items(self) -> List[Tuple[str, Any]]:
        """
        Return all data of dict. (key, value)
        """
        for k, v in zip(self.keys(), self.values()):
            yield k, v

    def remove(self, key):
        """
        Clear key and value.
        """
        self.del_default(key)

    def pop(self, key):
        """
        Clear key and return it's value.
        """
        value = self.get_default(key)
        self._del_default(key)

        if key in self:
            value = self.__getitem__(key)
            self.__delitem__(key)
        return value

    def popitem(self):
        """
        Pop the last item.
        :return:
        """
        k, v = super().__getattribute__(f"_{self.__class__.__name__}__default").popitem()
        if k in self:
            v = super().pop(k)
        return k, v

    def copy(self):
        _copy = DefaultDict(super().__getattribute__(f"_{self.__class__.__name__}__default"))
        _copy.update(self)
        return _copy


class TypedDict(AttrDict):
    """
    Dict only access one type for all element.
    Raise TypeOfValueError if type of set value not same as type defined before.

    Example:
        your_dict = UniqueTypeDict(int)
        your_dict.a = 1     # it's working
        your_dict.a = 2.0   # raise error TypeOfValueError

    # Un-support decorator
    """

    def __init__(self, _type, _args=None, _kwargs=None):
        super().__init__()
        if _kwargs is None:
            _kwargs = {}

        if _args is None:
            _args = ()

        if type(_args) is list:
            _args = tuple(_args)

        assert type(_args) is tuple
        assert isinstance(_kwargs, dict)

        if not isclass(_type):
            raise TypeError(f"Only support type class. But got {_type}")

        self._setattr('__type', _type)
        self._setattr('__args', _args)
        self._setattr('__kwargs', _kwargs)

    def add(self, key, value=None, force=False):
        if key in self and not force:
            raise KeyError(f'Key "{key}" was existed!')
        self.__setitem__(key, value)

    def __setitem__(self, key, value=None):
        if value is None:
            value = self.type(*self.__getattribute__(f"_{self.__class__.__name__}__args"),
                              **self.__getattribute__(f"_{self.__class__.__name__}__kwargs"))

        if not isinstance(value, self.type):
            try:
                value = self.type(value)
            except ValueError as e:
                e.args = f"Default is {self.type}. Got {type(value)}",
                raise e
        super().__setitem__(key, value)

    @property
    def type(self):
        return self.__getattribute__(f"_{self.__class__.__name__}__type")

    def setdefault(*args, **kwargs):
        raise AttributeError

    def default_params(self):
        raise AttributeError

    @staticmethod
    def fromkeys(*args, **kwargs):
        raise AttributeError

    def set_type(self, _type):
        if not isclass(_type):
            raise TypeError(f"Only support type class. But got {_type}")

        _values = list(self.values())
        for idx, _value in enumerate(_values):
            _values[idx] = _type(_value)

        super().setdefault('__type', _type)
        for k, v in zip(self.keys(), _values):
            self[k] = v


class DefaultTypeDict(DefaultDict):
    """
    DefaultTypeDict help cover your dict when set item same as type of default value.
    Implement from DefaultDict.

    Example:
        your_dict = DefaultTypeDict(a=int, b=float)
        your_dict.a = 1     # It's working
        your_dict.a = "default"   # Error TypeOfValueError

    Custom class:
        class abc(DefaultTypeDict):
            a: 1
            b: float

        your_dict = abc()
        your_dict.a = 1     # It's working
        your_dict.a = "default"   # Error TypeOfValueError

    ==================
    Supported decorator.
        @attrdict.default_type_dict

    Decorator that it create DefaultTypeDict base on attribute of class.
    Raise 'TypeError': On the same key, if type of value isn't same as type in annotations.

    Support attribute alias:
        @attrdict.default_type_dict
        class CustomDict:
            a: float = 1        # value will be cast to type that it was defined in annotation.
            b = 2               # if type not in annotation, type of value will be used.
            c: int = 'abcd'     # raise TypeError

        # assume key 'c' wasn't set.
        custom_dict = CustomDict()
        custom_dict.a   # return: 1.0
        custom_dict.b   # return: 2
    """
    def setdefault(self, _k=None, _v=None, **kwargs):
        if _k:
            kwargs.update({_k: _v})
        for k, v in kwargs.items():
            super().setdefault(k, v)
            _type = v if isclass(v) else v.__class__
            if k in self and not isinstance(self[k], _type):
                self[k] = _type(self[k])

    def __setitem__(self, key, value):
        if key not in super().__getattribute__(f"_{self.__class__.__name__}__default"):
            raise KeyError(f"Not found '{key}'")
        _type = self.get_default(key)

        if not isclass(_type):
            _type = _type.__class__

        if not isinstance(value, _type):
            try:
                value = _type(value)
            except ValueError as e:
                e.args = f"Default is {_type}. Got {type(value)}",
                raise e
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isclass(value):
            return value()
        return value

    def __setattr__(self, key, value):
        if key not in super().__getattribute__(f"_{self.__class__.__name__}__default"):
            raise KeyError(f"Not found '{key}'")

        _type = self.get_default(key)
        if not isclass(_type):
            _type = _type.__class__

        if not isinstance(value, _type):
            try:
                value = _type(value)
            except ValueError as e:
                e.args = f"Default is {_type}. Got {type(value)}",
                raise e
        return super().__setattr__(key, value)

    def __getattr__(self, key):
        value = super().__getattr__(key)
        if isclass(value):
            return value()
        return value


__all__ = ['DefaultDict', 'DefaultTypeDict', 'TypedDict']
