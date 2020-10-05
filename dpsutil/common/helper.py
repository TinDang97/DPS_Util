def check_type_instance(_instance, _types):
    if not isinstance(_instance, _types):
        raise TypeError(f"Required {_types}. Got {type(_instance)}")
    return True


def convert_kwargs_to_cmd_line_args(kwargs):
    """
    Helper function to build command line arguments out of dict.
    """
    args = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        args.append('-{}'.format(k))
        if v is not None:
            args.append('{}'.format(v))
    return args


def get_list_attribute(_object):
    """
    Return value list without built-in attribute.
    """
    return [v for k, v in _object.__dict__.items() if not k.startswith("__")]
