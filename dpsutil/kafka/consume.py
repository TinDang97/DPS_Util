from confluent_kafka import Consumer as _ConsumerImpl, TopicPartition
from confluent_kafka.cimpl import Message
from confluent_kafka.error import KafkaException

from .error import ConsumeError, TimeOutError, KeyDeserializationError, ValueDeserializationError
from .configs import ConsumerConfigs, _STOP_CODE, _MAX_MSG

__all__ = [
    'Consumer'
]


class MessageFetcher(object):
    """
    Fetch message and auto commit.

    Parameters
    ----------
    poll_func: function
        Function poll message from broker

    commit_func: function
        Function commit message

    max_message: int
        Maximum num of message to be returned

    raise_exception: bool
        If False, this iter will skip error message.
    """

    def __init__(self, poll_func, commit_func, max_message=-1, raise_exception=True):
        self.poll_func = poll_func
        self.commit_func = commit_func
        self.raise_exception = raise_exception
        self.max_message = max_message
        self.count_msg = 0
        self.prev_msg = None

    def __next__(self):
        if self.prev_msg is not None:
            self.commit_func(self.prev_msg)

        # check count message, infinity if max_message attribute is -1.
        if 0 <= self.max_message <= self.count_msg:
            raise StopIteration

        # fetch message from broker
        if self.raise_exception:
            msg = self.poll_func()
        else:
            # skip error message if raise_exception is False.
            while True:
                try:
                    msg = self.poll_func()
                    break
                except KafkaException as e:
                    if hasattr(e, 'message') and isinstance(e.message, Message):
                        self.commit_func(e.message)
                    raise e

        # None - stop event from Producer.
        if msg is None:
            raise StopIteration

        # count message
        self.count_msg += 1

        # commit message
        if self.count_msg % 500 == 0:
            self.commit_func(msg)

        return msg

    __enter__ = __iter__ = (lambda self: self)


class Consumer(_ConsumerImpl):
    """
    Consumer with deserializer key/value.

    Auto commit if KafkaException wasn't raised during fetch message.

    Parameters
    ----------
    group_id: str
        Group name to consumer join into queue task.

    topics: list|str
        Listening topic.

    raise_error: bool
        raise KafkaError during execute if True. Else silent.

    configs: dict
        configs of consumer. Ref: ConsumerConfigs

    raise_error: bool
        if False, 'func:consume and func:pool' skip error message.

    Raises
    ------
    KafkaException
        configs isn't supported.
    """

    def __init__(self, group_id, topics=None, configs=None, raise_error=True):
        if configs is None:
            configs = {}

        self.raise_error = raise_error

        configs = ConsumerConfigs({"group.id": group_id}, **configs)
        self._value_deserializer = configs.pop('value.deserializer')
        self._key_deserializer = configs.pop('key.deserializer')
        super().__init__(configs)

        if topics:
            self.subscribe(topics)

    def _deserializer_msg(self, msg: Message):
        if self._value_deserializer:
            try:
                msg.set_value(self._value_deserializer(msg.value()))
            except Exception as se:
                raise ValueDeserializationError(exception=se, message=msg)

        if self._key_deserializer:
            try:
                msg.set_key(self._key_deserializer(msg.key()))
            except Exception as se:
                raise KeyDeserializationError(exception=se, message=msg)
        return msg

    def subscribe(self, topics, on_assign=None, on_revoke=None, *args, **kwargs):
        """
        Set subscription to supplied list of topics
        This replaces a previous subscription.

        Regexp pattern subscriptions are supported by prefixing the topic string with ``"^"``, e.g.::

        consumer.subscribe(["^my_topic.*", "^another[0-9]-?[a-z]+$", "not_a_regex"])

        Parameters
        ----------
        topics: str|list[str]
            Topics want to consume.

        on_assign: function
            callback to provide handling of customized offsets on completion of a successful partition re-assignment.

        on_revoke: function
            callback to provide handling of offset commits to a customized store on the start of a rebalance operation.
        """
        if not isinstance(topics, list):
            topics = [topics]

        if on_assign is None:
            on_assign = (lambda err, msg: None)

        if on_revoke is None:
            on_revoke = (lambda err, msg: None)

        return super(Consumer, self).subscribe(topics, on_assign=on_assign, on_revoke=on_revoke, *args, **kwargs)

    def poll(self, timeout=-1):
        """
        Poll message from broker.

        Parameters
        ----------
        timeout: int
            Maximum time to block waiting for message, event or callback. (default: infinite (-1)). (Seconds)

        Returns
        -------
        None
            If event stop message from producer was received.
            raise RuntimeError if recall this methods.

        Message
            Message of topic which was consumed.

        Raises
        ------
        RuntimeError
            if called on a closed consumer

        TimeOutError
            if in timeout

        ConsumeError
            Raise message's error.
        """
        msg = super().poll(timeout=timeout)

        if msg is None:
            raise TimeOutError

        if msg.error():
            raise ConsumeError(msg.error().code(), message=msg)

        msg = self._deserializer_msg(msg)
        if msg.key() == _STOP_CODE:
            is_successed = self.commit(msg, asynchronous=False)[0]
            if is_successed.error:
                raise RuntimeError(f"Commit error. {is_successed.error}")
            self.close()
            return
        return msg

    def commit(self, message=None, offsets=None, asynchronous=True, *args, **kwargs):
        """
        Commit a message or a list of offsets.

        Parameters
        ----------
        message: Message
            Commit message's offset+1

        offsets: list[TopicPartition]
            List of TopicPartition to commit.

        asynchronous: bool
            Asynchronous commit, return None immediately.
            If False the commit() call will block until the commit succeeds

        Returns
        -------
        list[TopicPartition] | None
            None if asynchronous is True.

        Raises
        ------
        KafkaException
            in case of internal error

        RuntimeError
            if called on a closed consumer
        """
        super(Consumer, self).commit(message=message, offsets=offsets, asynchronous=asynchronous)

    def consume(self, num_messages=1, timeout=-1, *args, **kwargs):
        """
        Yield msg with limited quantity.

        Parameters
        ----------
        num_messages: int
            Maximum number of messages to return (default: 1).

        timeout: float
            Maximum time to block waiting for message, event or callback (default: infinite (-1)). (Seconds)

        Returns
        -------
        list[Message]
            A list of Message objects (possibly empty on timeout)

        Raises
        ------
        RuntimeError
            if called on a closed consumer

        ValueError
            if num_messages > 16777216 (2^24)
        """
        if num_messages > _MAX_MSG:
            raise ValueError("num_messages must smaller than 2^24 (16777216) messages.")

        msg_fetcher = MessageFetcher(self.poll, self.commit, max_message=num_messages, raise_exception=self.raise_error)
        return list(msg_fetcher)

    def __iter__(self):
        msg_fetcher = MessageFetcher(self.poll, self.commit, raise_exception=self.raise_error)
        return iter(msg_fetcher)
