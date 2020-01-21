import hashlib
import datetime


def short_hash(msg):
    return hashlib.sha1(msg.encode("UTF-8")).hexdigest()[:10]

def hash_now():
    return short_hash(datetime.datetime.now().isoformat())


