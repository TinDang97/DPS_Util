from dpsutil.kafka.consume import Consumer
from dpsutil.kafka.producer import Producer
import json
import numpy


def main():
    consumer = Consumer('abd', topics=['abcd'], raise_error=False, configs={
        "value.deserializer": json.loads
    })

    # producer = Producer()
    # producer.produce('abd', json.dumps({'a': 123}))

    for msg in consumer:
        print(msg.value())


if __name__ == '__main__':
    main()
