# DPS_Util
This repository contain all util.

## Install:

Get at https://pypi.org/project/dpsutil/

```
pip install dpsutil
```


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

- Implemented numpy.memmap with High Performance and control memory IO.

### Media:

Implemented OpenCV with:

- TurboJPEG (https://github.com/lilohuang/PyTurboJPEG)
- FFmpeg(https://github.com/kkroening/ffmpeg-python) 

To: improve read & write (image, video) IO speed. Faster than 2.6x OpenCV IO

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

## Issue:

#### Cmake error during install blosc

Firstly, you need install scikit

    $ pip install scikit-build
    
After that, follow instuction to install Cmake: 

https://cliutils.gitlab.io/modern-cmake/chapters/intro/installing.html

Python 

    $ pip install cmake>=3.12   

#### Not found 'FFmpeg':

Find FFmppeg lib at: https://www.ffmpeg.org/download.html

**Linux**:

    sudo apt-get install ffmpeg

#### Not found 'libjpeg-turbo':

Find FFmppeg lib at: https://libjpeg-turbo.org/Documentation/OfficialBinaries

**Linux**:

    $ sudo apt install libturbojpeg

#### Not found Redis or Kafka server:

- Make sure your Redis or Kafka server started.
- Make sure correct username & password.

#### Not found 'Could NOT find OpenSSL':

SSL development libraries have to be installed
https://github.com/openssl/openssl#build-and-install

Ubuntu:

    $ apt-get install libssl-dev libffi-dev

   

Develop by: TinDang\
OS: Ubuntu Bionic 18.4\
Python: 3.6.9 stable
