import ssl

from confluent_kafka import Producer, Consumer, SerializingProducer, DeserializingConsumer
import kafka

# TODO | Kafka Confluent. https://github.com/confluentinc/confluent-kafka-python

# Implement it with Optimize Kafka Config Tutorial |
# https://assets.confluent.io/m/6b6d4f8910691700/original/20190626-WP-Optimizing_Your_Apache_Kafka_Deployment.pdf?_ga=2.212964180.1913133577.1587956580-480563997.1585304727


def initial_consumer(*topic, bootstrap_servers='localhost', group_id=None, auto_offset_reset="earliest",
                     enable_auto_commit=True, sasl_plain_username=None, sasl_plain_password=None,
                     consumer_timeout_ms=float('inf'), value_deserializer=None):
    consumer = Consumer(
        dict(
            {"bootstrap.servers": bootstrap_servers},
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
    )
    consumer.subscribe(topic)
    return consumer


def initial_producer(bootstrap_servers='localhost', compression_type="lz4",
                     sasl_plain_username=None, sasl_plain_password=None, value_serializer=None,
                     max_request_size=1 * 1024 ** 2):
    sasl_mechanism, security_protocol, context = initial_ssl(sasl_plain_username)
    return Producer(
        {"bootstrap.servers": bootstrap_servers,
         "compression_type": compression_type,
         # "sasl.username": sasl_plain_username,
         # "sasl.password": sasl_plain_password,
         # "security.protocol": security_protocol
    })
    #     sasl_plain_username=sasl_plain_username,
    #     sasl_plain_password=sasl_plain_password,
    #     security_protocol=security_protocol,
    #     ssl_context=context,
    #     sasl_mechanism=sasl_mechanism,
    #     value_serializer=value_serializer,
    #     max_request_size=max_request_size
    # )
