from confluent_kafka.cimpl import KafkaException, KafkaError
from confluent_kafka.serialization import SerializationError
from confluent_kafka.cimpl import Message


class TimeOutError(KafkaException):
    """
    Raise error if timeout error 'py:cosumer.poll', 'py:cosumer.consume', 'py:producer.flush', 'py:producer.poll'

    Parameters
    ----------
        exception: Exception
           The original exception

        message: Message
           The Kafka Message returned from the broker.
    """
    def __init__(self, exception=None, message=None):
        code = KafkaError._TIMED_OUT
        if exception is not None:
            kafka_error = KafkaError(code, repr(exception))
            self.exception = exception
        else:
            kafka_error = KafkaError(code)
        super().__init__(kafka_error)
        self.__message = message
        self.__error = kafka_error

    @property
    def message(self):
        return self.__message

    @property
    def error(self):
        return self.__error


class ConsumeError(KafkaException):
    """
    Wraps all errors encountered during the consumption of a message.

    Note:
        In the event of a serialization error the original message contents
        may be retrieved from the ``message`` attribute.

    Parameters
    ----------
        error_code: int
            Error code indicating the type of error.

        exception: Exception
            The original exception

        message: Message
            The Kafka Message returned from the broker.
    """
    def __init__(self, error_code, exception=None, message=None):
        if exception is not None:
            kafka_error = KafkaError(error_code, repr(exception))
            self.exception = exception
        else:
            kafka_error = KafkaError(error_code)

        super(ConsumeError, self).__init__(kafka_error)
        self.__message = message
        self.__error = kafka_error

    @property
    def message(self):
        return self.__message

    @property
    def error(self):
        return self.__error


class KeyDeserializationError(ConsumeError, SerializationError):
    """
    Wraps all errors encountered during the deserialization of a Kafka
    Message's key.

    Parameters
    ----------
        exception: Exception
            The original exception

        message: Message
            The Kafka Message returned from the broker.
    """
    def __init__(self, exception=None, message=None):
        super(KeyDeserializationError, self).__init__(
            KafkaError._KEY_DESERIALIZATION, exception=exception, message=message)


class ValueDeserializationError(ConsumeError, SerializationError):
    """
    Wraps all errors encountered during the deserialization of a Kafka
    Message's value.

    Parameters
    ----------
        exception: Exception
            The original exception

        message: Message
            The Kafka Message returned from the broker.
    """
    def __init__(self, exception=None, message=None):
        super(ValueDeserializationError, self).__init__(
            KafkaError._VALUE_DESERIALIZATION, exception=exception, message=message)


class ProduceError(KafkaException):
    """
    Wraps all errors encountered when Producing messages.

    Parameters
    ----------
        error_code: int
            Error code indicating the type of error.

        exception: Exception
            The original exception
    """
    def __init__(self, error_code, exception=None):
        if exception is not None:
            kafka_error = KafkaError(error_code, repr(exception))
            self.exception = exception
        else:
            kafka_error = KafkaError(error_code)
        super(ProduceError, self).__init__(kafka_error)
        self.__error = kafka_error

    @property
    def error(self):
        return self.__error


class KeySerializationError(ProduceError, SerializationError):
    """
    Wraps all errors encountered during the serialization of a Message key.

    Parameters
    ----------
        exception: Exception
            The original exception
    """
    def __init__(self, exception=None):
        super(KeySerializationError, self).__init__(
            KafkaError._KEY_SERIALIZATION, exception=exception)


class ValueSerializationError(ProduceError, SerializationError):
    """
    Wraps all errors encountered during the serialization of a Message value.

    Parameters
    ----------
        exception: Exception
            The original exception
    """
    def __init__(self, exception=None):
        super(ValueSerializationError, self).__init__(
            KafkaError._VALUE_SERIALIZATION, exception=exception)
