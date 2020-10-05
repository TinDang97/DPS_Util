__author__ = "Tin Dang"
__copyright__ = "Copyright (C) 2020 DPS"
__license__ = "Public Domain"
__version__ = "1.0"
__doc__ = "Options builder like a property"

import inspect

from dpsutil.attrdict import AttrDict

SETFUNC_OPTION_TYPE = (bool, type(None), staticmethod, classmethod)


class InterruptedSetOption(Exception):
    pass


class NotSetOption(Exception):
    def __init__(self):
        super(NotSetOption, self).__init__("The option hasn't been set.")
        self.__traceback__ = None

    def __bool__(self):
        return False

    def __eq__(self, other):
        return isinstance(other, NotSetOption)


class Option(property):
    """
    Define options as property.
    Implement from property python built-in

    Parameters
    ----------
    name: str | None
        Key of option. Auto fill if not set.

    set_filter: function | None | bool
        Filter value before set.

    doc: str
        docstring
    """

    def __init__(self, name, set_filter=None, doc=None):
        self.__name = name
        self.__doc = doc

        if not hasattr(set_filter, "__call__"):
            if not isinstance(set_filter, SETFUNC_OPTION_TYPE):
                raise TypeError(f"Set filter must in {SETFUNC_OPTION_TYPE}. Got {type(set_filter)}")

            elif isinstance(set_filter, (staticmethod, classmethod)):
                set_filter = set_filter.__func__

        if hasattr(set_filter, "__call__"):
            num_args = len(inspect.getfullargspec(set_filter).args)
            if num_args == 0:
                wrap_set_filter = (lambda *args, **kwargs: set_filter())
            elif num_args == 1:
                wrap_set_filter = (lambda value, *args, **kwargs: set_filter(value))
            else:
                wrap_set_filter = (lambda _self, value, *args, **kwargs: set_filter(_self, value))
        elif isinstance(set_filter, bool) and not set_filter:
            wrap_set_filter = ValueError("Read-only options.")
        else:
            wrap_set_filter = None

        def _wrap_setter(_self, value):
            if hasattr(wrap_set_filter, "__call__"):
                try:
                    value = wrap_set_filter(_self=_self, value=value)
                except InterruptedSetOption:
                    return
            elif isinstance(wrap_set_filter, Exception):
                raise wrap_set_filter
            return OptionsBase.set(_self, self.__name, value)

        def _wrap_getter(_self):
            return OptionsBase.get(_self, self.__name)

        def _wrap_deleter(_self):
            return OptionsBase.delete(_self, self.__name)

        super().__init__(_wrap_getter, _wrap_setter, _wrap_deleter)

    @property
    def name(self):
        return self.__name

    @property
    def doc(self):
        return self.__doc


class OptionsBase(object):
    def get_option_name(self, attr):
        return getattr(self.__class__, attr).name

    def get(self, name):
        if name not in self.__opts:
            raise NotSetOption
        return self.__opts.__getitem__(name)

    def set(self, name, value):
        self.__opts.__setitem__(name, value)

    def delete(self, name):
        if name not in self.__opts:
            raise NotSetOption
        self.__opts.__delitem__(name)

    def is_existed(self, name):
        if isinstance(name, Option):
            name = name.name
        return self.__opts.__contains__(name)

    def clear(self):
        self.__opts.clear()

    def diff(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(f"Type `{type(other)}` can't compare.")

        diff = {}
        for k in set(self.__opts.keys()).difference(other.__opts):
            diff[k] = self.__opts[k]

        for k in set(other.__opts.keys()).difference(self.__opts):
            diff[k] = other.__opts[k]
        return diff

    def from_options(self, _options):
        if not isinstance(_options, OptionsBase) or not isinstance(self, type(_options)):
            raise TypeError(f"Type `{type(_options)}` can't get options.")
        self.__opts.update(_options.__opts)

    def list_options(self):
        options = list()
        for attr in dir(self):
            if attr.startswith("__"):
                continue

            try:
                opt = getattr(self.__class__, attr)
                if isinstance(opt, Option):
                    options.append((attr, opt.name, self[attr]))
            except NotSetOption:
                pass
        return options

    def __init__(self, options=None):
        self.__opts = AttrDict()

        if options:
            self.from_options(options)

    def __repr__(self):
        _str = ""
        for name, k, v in self.list_options():
            _str += f"\n\t\"{k}\" ({name}): "

            if isinstance(v, OptionsBase):
                _str += f"{v.__class__.__name__}"
                if v:
                    _str += f"({v.__str__()})"
                else:
                    _str += "(None)"
            else:
                doc = getattr(self.__class__, name)
                _str += f"{v}{f' | {doc.doc}' if doc.doc else ''}"
        if not _str:
            _str = "None"
        return f"{self.__class__.__name__}: {_str}\n"

    def __str__(self):
        return self.__opts.__str__()

    def __bool__(self):
        return bool(self.__opts)

    def __contains__(self, name):
        if isinstance(name, Option):
            name = name.name

        try:
            return hasattr(self, name)
        except NotSetOption:
            return True
        except AttributeError:
            return False

    def __setitem__(self, name, value):
        return self.__setattr__(name, value)

    def __getitem__(self, name):
        return self.__getattribute__(name)

    def __delitem__(self, name):
        return self.__delattr__(name)

    def __iter__(self):
        return sorted(self.__opts.keys())
    
    def __dict__(self):
        return dict(self.__opts)
    

class Options(OptionsBase):
    """
    Option Container.
    """
    def dict(self):
        return super(Options, self).__dict__()

    def get(self, *args, **kwargs):
        raise AttributeError

    def set(self, *args, **kwargs):
        raise AttributeError

    def delete(self, *args, **kwargs):
        raise AttributeError


def option(name, set_filter=None, doc=None):
    """
    Option decorator

    Examples
    --------
    >> class Test(Options):
        test = option("test")

    >> a = Test()

    >> a.test
    # None

    >> a.build()
    # []

    >> a.test = None

    >> a.build()
    # ['-test']

    >> a.test = 5

    >> a.build()
    # ['-test', 5]

    >> del a.test

    >> a.test
    # None

    >> a.build()
    # []
    """
    return Option(name, set_filter=set_filter, doc=doc)


def check_stacks(*funcs):
    def _func(_value):
        for func in funcs:
            _value = func(_value)
        return _value

    return _func


def check_min_value(_min):
    def _func(_value):
        check_type(_value, int)
        if _value < _min:
            raise ValueError(f"Value must >= {_min}. MIN: {_min}")
        return _value

    return _func


def check_max_value(_max):
    def _func(_value):
        check_type(_value, int)
        if _value > _max:
            raise ValueError(f"Value must <= {_max}. MAX: {_max}")
        return _value

    return _func


def check_in_range(_min, _max):
    return check_stacks(check_min_value(_min), check_max_value(_max))


def is_not_params(_value):
    if _value is not None:
        raise ValueError("No require parameters. Set value is `None` to active option.")
    return _value


def value_in_list(_list):
    def _func(_value):
        if _value not in _list:
            raise ValueError(f'Value must in {_list}. But got "{_value}"')
        return _value

    return _func


def check_type(*_type):
    def _func(_value):
        if not isinstance(_value, _type):
            raise ValueError(f"Value's type must be {_type}. But got \"{type(_value)}\"")
        return _value

    return _func
