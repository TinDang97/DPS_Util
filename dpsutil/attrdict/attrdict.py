import pickle
import attrdict


class KeyNotFound(Exception):
    pass


class AttrDict(attrdict.AttrDict):
    """
    AttrDict will help you get value of key via attributes. Implement attrdict.AttrDict
    More info here: https://github.com/bcj/AttrDict

    Example:
        your_dict = AttrDict(a=1, b=2)
        your_dict.a     # return: 1
        your_dict.b     # return: 2
    """
    def __getitem__(self, item):
        return super().get(item)

    def __getattr__(self, item):
        return super().get(item)

    def __repr__(self):
        return dict(self).__repr__()

    def copy(self):
        return AttrDict(self)


__all__ = ['AttrDict', 'KeyNotFound']
