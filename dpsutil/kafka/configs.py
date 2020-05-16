from dpsutil.attrdict import DefaultDict, AttrDict


# https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
# https://assets.confluent.io/m/6b6d4f8910691700/original/20190626-WP-Optimizing_Your_Apache_Kafka_Deployment.pdf?_ga=2.212964180.1913133577.1587956580-480563997.1585304727


__all__ = [
    "_STOP_CODE",
    "_MAX_MSG",
    "sasl_conf",
    "ConsumerConfigs",
    "ProducerConfigs"
]

_STOP_CODE = b"__CONSUMER_STOP"
_MAX_MSG = 1 << 24


def sasl_conf(sasl_mechanism="GSSAPI", username=None, password=None, broker_principal="kafka"):
    """
    SASL supported mechanisms: GSSAPI, PLAIN, SCRAM-SHA-256, SCRAM-SHA-512
    """
    sasl = {
        'sasl.mechanism': sasl_mechanism.upper(),
        # Set to SASL_SSL to enable TLS support.
        'security.protocol': 'SASL_PLAINTEXT'
    }

    if sasl_mechanism != 'GSSAPI':
        sasl.update({'sasl.username': username,
                     'sasl.password': password})

    if sasl_mechanism == 'GSSAPI':
        sasl.update({'sasl.kerberos.service.name': broker_principal,
                     # Keytabs are not supported on Windows. Instead the
                     # the logged on user's credentials are used to
                     # authenticate.
                     'sasl.kerberos.principal': username,
                     'sasl.kerberos.keytab': password})
    return sasl


DEFAULT_COMMON = AttrDict({
    "bootstrap.servers": "localhost:9092",
    "message.max.bytes": 2621440,  # 2.5 mb
    # **sasl_conf(),
})

DEFAULT_CONSUMER = AttrDict({
    "group.id": None,
    "value.deserializer": None,
    "key.deserializer": None,
    "fetch.min.bytes": 100000,
    "enable.auto.commit": True,
    "auto.commit.interval.ms": 5000,
    "auto.offset.reset": 'latest',  # smallest, earliest, beginning, largest (default), latest, end, error
    "session.timeout.ms": 6000,  # default 10000 (10s)
    "heartbeat.interval.ms": 2000,  # default 3000 (3s)

})

DEFAULT_PRODUCER = AttrDict({
    "linger.ms": 20,  # 0 (default) in kafka. 10-100 (optimize for throughput)
    "compression.type": "lz4",  # none (default) in kafka. LZ4 (optimize for throughput)
    "compression.level": 1,  # default level: -1, .
    "acks": 1,  # default: 1
    "value.serializer": None,
    "key.serializer": None,
})


class ConsumerConfigs(DefaultDict):
    def __init__(self, *args, **configs):
        configs = AttrDict(*args, **configs)
        super().__init__(DEFAULT_COMMON, **DEFAULT_CONSUMER)
        self.setdefault(configs)


class ProducerConfigs(DefaultDict):
    def __init__(self, *args, **configs):
        configs = AttrDict(*args, **configs)
        super().__init__(DEFAULT_COMMON, **DEFAULT_PRODUCER)
        self.setdefault(configs)
