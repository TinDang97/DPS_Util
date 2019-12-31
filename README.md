# DPS_Util
This repository contain all util like compression, opencv2 toolkit,...

## Todo:
- ~~Add compression~~
- ~~Add KafkaWrapper~~
- ~~Add RedisWrapper~~
- Add opencv toolkit

## Task Done:

### Compression Lossless:
- Support type: ndnumpy, dict, string, float, int
- Compress by lzma
- if data's type is ndnumpy, compressor will dump by Pickle (support only Python)
otherwise json

### KafkaWrapper:
Wrapping Consumer and Producer with default setting and security.

### RedisWrapper
Wrapping Redis Connector with default setting and security.