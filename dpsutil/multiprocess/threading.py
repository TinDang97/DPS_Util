from collections import deque
from queue import Queue


class QueueOverflow(Queue):
    def _init(self, maxsize: int) -> None:
        self.queue = deque(maxlen=maxsize)

    def put(self, item, *args, **kwargs):
        self.put_nowait(item)

    def put_nowait(self, item):
        with self.not_full:
            self._put(item)
            self.not_empty.notify()
