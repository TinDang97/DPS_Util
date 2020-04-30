import numpy

import dpsutil.media.image as image_tool

# TODO: implement to easier using.


class Transform(object):
    def __init__(self, methods):
        if not isinstance(methods, (list, tuple)):
            raise TypeError("Methods must be a list.")
        for method in methods:
            if hasattr(method, "__call__"):
                continue
            raise AttributeError(f"{method.__name__ if hasattr(method, '__name__') else method.__class__.name} "
                                 f"hasn't __call__ attribute!")
        self.methods = methods

    def perform(self, img):
        for method in self.methods:
            img = method(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.methods:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

    __call__ = perform


class TransformMethod(object):
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        method, args, kwarg = self.function(*args, **kwargs)

        def runner(img):
            return method(img, *args, **kwargs)

        return runner


@TransformMethod
def crop(box, *args, **kwargs):
    return image_tool.crop, [box, *args], kwargs


@TransformMethod
def resize(size, *args, keep_ratio=False, inter_method=image_tool.INTER_DEFAULT, **kwargs):
    return image_tool.resize, [size, *args], dict(keep_ratio=keep_ratio, inter_method=inter_method, **kwargs)


@TransformMethod
def crop_center(crop_size, *args, **kwargs):
    return image_tool.crop_center, [crop_size, *args], kwargs


@TransformMethod
def crop_margin(margin_size, box, *args, **kwargs):
    return image_tool.crop_margin, [margin_size, *args], {'box': box, **kwargs}


@TransformMethod
def flip(flip_mode, *args, **kwargs):
    return image_tool.flip, [flip_mode, *args], kwargs


@TransformMethod
def normalize(mean, std, *args, **kwargs):
    return image_tool.normalize, [mean, std, *args], kwargs
