from confluent_kafka import Producer as _ProducerImpl

from .configs import ProducerConfigs, _STOP_CODE
from .error import ValueSerializationError, KeySerializationError, TimeOutError

__all__ = [
    'Producer'
]


class Producer(_ProducerImpl):
    """
    Synchronous/Asynchronous Kafka Producer with serializer key/value
    """

    def __init__(self, configs=None, produce_topics=None):
        if configs is None:
            configs = {}

        self.__produce_topic = []

        if produce_topics is not None:
            self.add_topics(produce_topics)

        configs = ProducerConfigs(configs)
        self._value_serializer = configs.pop('value.serializer')
        self._key_serializer = configs.pop('key.serializer')

        super().__init__(configs)

    def _serializer_msg(self, key, value):
        if self._value_serializer:
            try:
                value = self._value_serializer(value)
            except Exception as se:
                raise ValueSerializationError(se)

        if self._key_serializer:
            try:
                key = self._key_serializer(key)
            except Exception as se:
                raise KeySerializationError(se)
        return key, value

    def add_topics(self, topics):
        """
        Add produce topic list.

        Parameters
        ----------
        topics: str|list
            Topic or topic's list, which will be produce by default.
        """

        if not isinstance(topics, list):
            topics = [topics]

        for topic in topics:
            if not isinstance(topic, str):
                raise TypeError(f"Expected: topic's type must be str. But got {type(topic)}")

            if topic in self.__produce_topic:
                continue

            self.__produce_topic.append(topic)

    def clear_topics(self):
        """
        Remove all produce topic.
        """
        return self.__produce_topic.clear()

    def stop_msg(self, topics=None):
        """
        Send stop event to topic. old_topic_produced will be sent if topic isn't given

        Parameters
        ---------
        topics: list|str
            Topics will be send STOP_EVENT.
            If None, all topics was recorded that used.
        """
        if topics is not None and not isinstance(topics, list):
            topics = [topics]

        if topics is None:
            topics = self.__produce_topic

        self.produce(topics, key=_STOP_CODE)

    def produce_async(self, topics=None, value=None, key=None, headers=None, partition=-1, on_delivery=None,
                      timestamp=0, record_topics=True):
        """
        Async produce message to topics.
        """
        return self.produce(
            topics=topics, value=value, key=key, headers=headers, partition=partition,
            on_delivery=on_delivery, timestamp=timestamp, record_topics=record_topics,
            block=False
        )

    def produce(self, topics=None, value=None, key=None, headers=None, partition=-1, on_delivery=None, timestamp=-1,
                record_topics=True, block=True, timeout=-1, *args, **kwargs):
        """
        Produce message to topics.

        Parameters
        ----------
        topics: str|list[str]
            Topic to produce message to.
            If None, all topics was recorded that used.

        value: str|bytes
            Message payload

        key: str|bytes
            Message key

        headers: dict|list
            Message headers to set on the message.
            The header key must be a string while the value must be binary, unicode or None.
            Accepts a list of (key,value) or a dict. (Requires librdkafka >= v0.11.4 and broker version >= 0.11.0.0)

        partition: int
            Partition to produce to, else uses the configured built-in partitioner.

        on_delivery: function
            Delivery report callback to call (from :py:func:`poll()` or :py:func:`flush()`)
            on successful or failed delivery

        timestamp: int
            Message timestamp (CreateTime) in milliseconds since epoch UTC (requires librdkafka >= v0.9.4,
            api.version.request=true, and broker >= 0.10.0.0). Default value is current time.

        record_topics: bool
            Save topics into topic list.

        block: bool
            Block to messages was delivered. (make producer synchronous)
            If on timeout, raise TimeOutError

        timeout: int
            Time (in seconds) waiting to delivery message.
            default(-1): infinity

        Raises
        ------
        BufferError
            if the internal producer message queue is full (``queue.buffering.max.messages`` exceeded)

        KafkaException
            for other errors, see exception code

        NotImplementedError
            if timestamp is specified without underlying library support.

        TimeOutError
            if block is True and flushing messages on timeout.
        """

        if topics is not None and not isinstance(topics, list):
            topics = [topics]
            if record_topics:
                self.add_topics(topics)

        if topics is None:
            topics = self.__produce_topic

        key, value = self._serializer_msg(key, value)

        if on_delivery is None:
            on_delivery = (lambda err, msg: None)

        for topic in topics:
            super().produce(topic, value=value, key=key, partition=partition, on_delivery=on_delivery,
                            timestamp=timestamp, headers=headers)

        if block:
            msg_unprocessed = self.flush(timeout=timeout)
            if msg_unprocessed > 0:
                raise TimeOutError
