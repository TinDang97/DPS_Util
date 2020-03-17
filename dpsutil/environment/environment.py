from ..attrdict import DefaultDict, KeyNotFound
from os import environ


class Environment(DefaultDict):
    """
    Environment: Auto pair environment parameters fit with default, which provided before.
    Implement attrdict.DefaultDict

    Example:
        env = Environment(HOME="./", DEFAULT_KEY="default_value")
        env            # {'HOME': '/home/user_name', 'DEFAULT_KEY': 'default_value'}

        # Get item with both lower and upper key.
        env.home       # '/home/user_name'
        env.HOME       # '/home/user_name'

    **Note: KEY must be upper or lower. Otherwise, raise KeyError

    Compare:
    # Regularly way. -> 491 characters

    configs = {
        'kafka_host': f'{os.environ.get('KAFKA_HOST', 'localhost')}:{os.environ.get('KAFKA_PORT', '9092')}',
        'kafka_user_name': os.environ.get('KAFKA_USER_NAME'),
        'kafka_password': os.environ.get('KAFKA_PASSWORD'),
        'redis_host': f'{os.environ.get('REDIS_HOST', 'localhost')}:{os.environ.get('REDIS_PORT", '6379')}',
        'redis_password': os.environ.get('REDIS_PASSWORD'),
        'redis_expire_time': int(os.environ.get('REDIS_EXPIRE_TIME', 60))
    }

    # With Environment -> 193 characters

    configs = Environment(KAFKA_HOST='localhost', KAFKA_PORT='9092', KAFKA_USER_NAME=None, KAFKA_PASSWORD=None,
                          REDIS_HOST='localhost', REDIS_PORT='6379', REDIS_PASSWORD=None, REDIS_EXPIRE_TIME=60)

    """

    def __init__(self, **default_params):
        super().__init__()

        # Add default key, value
        for k, v in default_params.items():
            k = self._cvt_key(k)
            self.setdefault(k, v)
            self.__setitem__(k, v)

        # Update environ value fit with default_params.
        for k, v in environ.items():
            try:
                self.__setitem__(k, v)
            except KeyNotFound:
                pass

    @staticmethod
    def _cvt_key(key):
        if key.islower():
            key = key.upper()

        if not key.isupper():
            raise KeyError(f"Environment name must be {key.upper()} or {key.lower()}. But got: {key}")

        return key

    def __getitem__(self, key):
        return super().__getitem__(self._cvt_key(key))

    def __setitem__(self, key, value):
        if key not in self.default_params():
            raise KeyNotFound

        if type(value) != type(self.default_params()[key]) \
                and value is not None and self.default_params()[key] is not None:
            value = type(self.default_params()[key])(value)

        return super().__setitem__(self._cvt_key(key), value)

    def add(self, env, value):
        self.setdefault(env, value)
        self.__setitem__(env, environ.get(env))

    def to_lower(self):
        curr = self.copy()
        for k in list(curr.keys()):
            value = curr.pop(k)
            curr[k.lower()] = value

        return curr


__all__ = ['Environment']
