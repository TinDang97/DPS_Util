from redis import Redis


def initial_redis(host='localhost', port=6379, db=0, password=""):
    db = Redis(host=host, port=port, db=db, password=password)
    return db
