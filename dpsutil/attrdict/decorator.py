from .attrdict import AttrDict, ReadOnlyDict
from .defaultdict import DefaultDict, DefaultTypeDict, TypedDict
import inspect


__all__ = ['attribute_dict', 'default_dict', 'default_typed_dict', 'typed_dict']


def _get_annotations(_cls):
    """
    Yield (key, type, value) of annotations

    class Annotations(object):
        a: int = 1
        b: float = 2.

    list(_get_annotations(_cls))
    # return: [('a', int, 1), ('b', float, 2)]
    """
    if hasattr(_cls, "__annotations__"):
        raise StopIteration

    for _key, _type in _cls.__annotations__:
        _value = getattr(_cls, _key, None)
        yield _key, _type, _value


def _get_vars_cls(_cls):
    """
    Yield all attributes and it's value which isn't in object class.

    class AnyClass(object):
        a=1
        b=2.

    list(_get_vars_cls(_cls))
    # return: [('a', 1), ('b', 2)]
    """
    if not inspect.isclass(_cls):
        raise TypeError(f"Expect class object. Got '{type(_cls)}'")

    keys = set(_cls.__dict__).difference(set(object.__dict__))
    for k in keys:
        if k.startswith("_"):
            continue
        yield k, _cls.__dict__[k]


def attribute_dict(_cls):
    """
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
    def instance():
        return AttrDict(_get_vars_cls(_cls))
    return instance


def readonly_dict(_cls):
    """
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
    def instance():
        return ReadOnlyDict(_get_vars_cls(_cls))
    return instance


def default_dict(_cls):
    """
    Decorator that it create DefaultDict by attribute of class.
    Support attribute alias:
        @attrdict.default_dict
        class CustomDict:
            a=1
            b=2

        custom_dict = CustomDict()
        custom_dict.a   # return: 1
        custom_dict.b   # return: 2
    """
    def instance():
        return DefaultDict(_get_vars_cls(_cls))
    return instance


def typed_dict(_cls):
    def instance():
        return TypedDict(_get_vars_cls(_cls))
    return instance


def default_typed_dict(_cls):
    """
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
    def instance():
        base_dict = DefaultTypeDict()
        annotations = getattr(_cls, '__annotations__', {})

        for _key, _value in _get_vars_cls(_cls):
            if _key in annotations:
                _type = annotations[_key]

                if not inspect.isclass(_type):
                    raise TypeError(f"'{_key}' in annotations must be class. Got '{_type}'")

                _value = _type(_value)
            base_dict.setdefault(_key, _value)

        for _key, _type, _value in _get_annotations(_cls):
            if _key in base_dict:
                continue
            base_dict.setdefault(_key, _value)
        return base_dict
    return instance
