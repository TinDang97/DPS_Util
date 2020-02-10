import hashlib
import datetime


def short_hash(msg: (str, bytes)) -> str:
    assert isinstance(msg, (str, bytes))
    if type(msg) is str:
        msg = msg.encode("UTF-8")
    return hashlib.sha1(msg).hexdigest()[:10]


def hash_now() -> str:
    return short_hash(datetime.datetime.now().isoformat())
