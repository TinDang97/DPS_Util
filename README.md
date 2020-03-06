# DPS_Util
This repository contain all util.

## Todo:

- Numpy Pool -> Cache Algorithm in RAM
- Sort -> implement natsort - https://github.com/SethMMorton/natsort
- CV -> Find more util
- Find & Add more functions

## Task Done:

### Compression Lossless:

- Support type: ndnumpy, bytes
- Compress by blosc. It support multi compressor and multi-thread.

### KafkaWrapper:

- Wrapping Consumer and Producer with default setting and security.

### RedisWrapper

- Wrapping Redis Connector with default setting and security.

### Numpy Pool:

- Implemented numpy.memmap with Hight Peromance

### Media:

Implemented OpenCV with TurboJPEG (https://github.com/lilohuang/PyTurboJPEG) 
and FFmpeg(https://github.com/kkroening/ffmpeg-python) 
-> Improve read & write (image, video) IO speed.

- Added some function which used frequently.
- More info: find in dpsutil.media

### Computer Vision (cv):

- Added Face Align with five landmark.

### Distance:

All function execute in numpy.
- Added cosine_similarity
- Added cosine
- Added euclidean_distance
- Added convert distance functions

### Other:

- Hashing