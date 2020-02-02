import ssl
from kafka import KafkaProducer, KafkaConsumer


def initial_ssl(username):
    if username is None:
        return None, 'PLAINTEXT', None

    sasl_mechanism = 'PLAIN'
    security_protocol = 'SASL_PLAINTEXT'
    ssl_context = ssl.create_default_context()
    ssl_context.options &= ssl.OP_NO_TLSv1
    ssl_context.options &= ssl.OP_NO_TLSv1_1
    return sasl_mechanism, security_protocol, ssl_context


def initial_consumer(topic, bootstrap_servers='localhost', group_id=None, auto_offset_reset="earliest",
                     enable_auto_commit=True, sasl_plain_username=None, sasl_plain_password=None,
                     consumer_timeout_ms=float('inf'), value_deserializer=None):
    sasl_mechanism, security_protocol, ssl_context = initial_ssl(sasl_plain_username)

    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=enable_auto_commit,
        sasl_plain_username=sasl_plain_username,
        sasl_plain_password=sasl_plain_password,
        security_protocol=security_protocol,
        ssl_context=ssl_context,
        sasl_mechanism=sasl_mechanism,
        value_deserializer=value_deserializer,
        consumer_timeout_ms=consumer_timeout_ms
    )


def initial_producer(bootstrap_servers='localhost', compression_type='lz4',
                     sasl_plain_username=None, sasl_plain_password=None, value_serializer=None,
                     max_request_size=1*1024**2):
    sasl_mechanism, security_protocol, context = initial_ssl(sasl_plain_username)
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        compression_type=compression_type,
        sasl_plain_username=sasl_plain_username,
        sasl_plain_password=sasl_plain_password,
        security_protocol=security_protocol,
        ssl_context=context,
        sasl_mechanism=sasl_mechanism,
        value_serializer=value_serializer,
        max_request_size=max_request_size
    )
