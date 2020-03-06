from redis import Redis


def initial_redis(host='localhost:6379', db=0, password=""):
    host, port = host.split(":")
    db = Redis(host=host, port=port, db=db, password=password)
    return db
