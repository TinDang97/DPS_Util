from dpsutil.attrdict import DefaultDict, AttrDict
from dpsutil.attrdict.decorator import _get_vars_cls
from os import environ


class Environment(DefaultDict):
    """
    Environment: Auto pair environment parameters fit with default, which provided before.
    Implement attrdict.DefaultDict

    Auto check and broadcast to default value's type.

    Example:
        env = Environment(HOME="./", DEFAULT_KEY="default_value")
        env            # {'HOME': '/home/user_name', 'DEFAULT_KEY': 'default_value'}

        # Get item with both lower and upper key.
        env.home       # '/home/user_name'
        env.HOME       # '/home/user_name'

    **Note: KEY must be upper or lower. Otherwise, raise KeyError

    Compare (without space character):
    # Regularly way. -> 433 characters

    configs = {
        'kafka_host': f'{os.environ.get('KAFKA_HOST', 'localhost')}:{os.environ.get('KAFKA_PORT', '9092')}',
        'kafka_user_name': os.environ.get('KAFKA_USER_NAME'),
        'kafka_password': os.environ.get('KAFKA_PASSWORD'),
        'redis_host': os.environ.get('REDIS_HOST', 'localhost'),
        'redis_port': os.environ.get('REDIS_PORT", '6379'),
        'redis_password': os.environ.get('REDIS_PASSWORD'),
        'redis_expire_time': int(os.environ.get('REDIS_EXPIRE_TIME', 60))
    }

    # With Environment -> 185 characters

    configs = Environment(KAFKA_HOST='localhost', KAFKA_PORT='9092', KAFKA_USER_NAME=None, KAFKA_PASSWORD=None,
                          REDIS_HOST='localhost', REDIS_PORT='6379', REDIS_PASSWORD=None, REDIS_EXPIRE_TIME=60)
    ==================
    Supported decorator.
        @environment.decorator

    Decorator create EnvDict base on attribute of class.

    @environment.env_decorator
    class CustomEnv:
        KAFKA_HOST = 'localhost'
        KAFKA_PORT = '9092'
        KAFKA_USER_NAME = None
        KAFKA_PASSWORD = None
        REDIS_HOST = 'localhost'
        REDIS_PORT = '6379'
        REDIS_PASSWORD = None
        REDIS_EXPIRE_TIME = 60
    """

    def __init__(self, *args, **default_params):
        super().__init__(*args, **default_params)

    def setdefault(self, _k=None, _v=None, **kwargs):
        if _k:
            kwargs.update({_k: _v})
        for k, v in kwargs.items():
            k = self._cvt_key(k)
            super().setdefault(k, v)
            if k in environ:
                super().__setitem__(k, environ[k])

    @staticmethod
    def _cvt_key(key):
        if key.islower():
            key = key.upper()

        if not key.isupper():
            raise KeyError(f"Environment name must be {key.upper()} or {key.lower()}. But got: {key}")
        return key

    def _cvt_value(self, key, value):
        if value is not None and self.get_default(key) is not None \
                and not isinstance(value, self.get_default(key).__class__):
            try:
                value = self.get_default(key).__class__(value)
            except ValueError:
                raise ValueError("Type of default is't same as set value. Change default before set.")
        return value

    def __getitem__(self, key):
        key = self._cvt_key(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        key = self._cvt_key(key)
        value = self._cvt_value(key, value)
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        raise AttributeError("Clear value doesn't supported. To get default value by get_default function.")

    def __getattr__(self, key):
        key = self._cvt_key(key)
        return super().__getattr__(key)

    def __setattr__(self, key, value):
        key = self._cvt_key(key)
        value = self._cvt_value(key, value)
        return super().__setattr__(value, value)

    def __delattr__(self, key, **kwargs):
        key = self._cvt_key(key)
        return super().__delattr__(key)

    def get_lower_dict(self):
        curr = AttrDict(self)
        for k in list(curr.keys()):
            value = curr.pop(k)
            curr[k.lower()] = value
        return curr


def env_decorator(_cls):
    """
    Decorator create EnvDict base on attribute of class.

    @environment.env_decorator
    class CustomEnv:
        KAFKA_HOST = 'localhost'
        KAFKA_PORT = '9092'
        KAFKA_USER_NAME = None
        KAFKA_PASSWORD = None
        REDIS_HOST = 'localhost'
        REDIS_PORT = '6379'
        REDIS_PASSWORD = None
        REDIS_EXPIRE_TIME = 60
    """
    def instance():
        return Environment(_get_vars_cls(_cls))
    return instance


__all__ = ['Environment', 'env_decorator']
