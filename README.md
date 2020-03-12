# DPS_Util
This repository contain all util.

### Install:

Get at https://pypi.org/project/dpsutil/

```
pip install dpsutil
```

### Change log:

Ver 1.0.0:
- Compression -> DONE
- KafkaWrapper -> DONE
- RedisWrapper -> DONE
- Vector -> DONE
- Media -> DONE
- Computer Vision -> DONE
- Distance -> DONE
- Hashing -> DONE

Ver 1.0.1:
- Added image_info -> Get (format, width, height) of image without decoding.
- Upgrade imdecode -> Auto decode image but need not provide encode_type.

Ver 1.0.2:
- Change '_numpy.pool_' to '_vector.pool_'
- Fix bugs

Ver 1.0.3:
- Compression:
    - Support list compression: _compress_list, decompres_list_
- Distance: 
    - Change distance.function -> vector.distance

Ver 1.1.0:
- Added AttrDict, FixedDict, FixedTypeDict

Ver 1.1.1:
- Fix bugs
- Added UniqueTypeDict

Ver 1.1.2:
- Fix bugs
- Changed module fixdict -> defaultdict
- Changed FixedDict -> DefaultDict
- Changed FixedTypeDict -> DefaultTypeDict

Ver 1.1.3:
- Fix bugs of attrdict.UniqueTypeDict

### Todo:

- Numpy Pool -> Cache Algorithm in RAM
- Sort -> implement natsort - https://github.com/SethMMorton/natsort
- CV -> Find more util
- Find & Add more functions
 ---
### Features

---
#### Compression Lossless:

- Support type: ndnumpy, bytes
- Compress by blosc. It support multi compressor and multi-thread.

#### KafkaWrapper:

- Wrapping Consumer and Producer with default setting and security.

#### RedisWrapper

- Wrapping Redis Connector with default setting and security.

#### Vector Pool:

- Implemented numpy.memmap with High Performance and control memory IO.

#### Media:

Implemented OpenCV with:

- TurboJPEG (https://github.com/lilohuang/PyTurboJPEG)
- FFmpeg(https://github.com/kkroening/ffmpeg-python) 

To: improve read & write (image, video) IO speed.
- Faster than OpenCV:
  - 2-6x in JPEG
  - 1.1x with others.

- Added some function which used frequently.
- More info: find in dpsutil.media

#### Computer Vision (cv):

- Added Face Align with five landmark.

#### Distance:

- Added cosine_similarity
- Added cosine
- Added euclidean_distance
- Added convert distance functions

_*Note: all function execute in numpy._

#### Attributes dict:

- **AttrDict**: will help you get value of key via attributes. Implement attrdict.AttrDict
- **FixedDict**: help cover your dict with (keys, values) that was defined before. Implement from AttrDict
- **FixedTypeDict**: help cover your dict when set item. Implement from FixedDict.
- **UniqueTypeDict**: dict only access one type for all element.

#### Other:

- Hashing
- Sort
---
### Issue:

---
#### Cmake error during install blosc

Firstly, you need install scikit

    $ pip install scikit-build
    
After that, follow instuction to install Cmake: 

> https://cliutils.gitlab.io/modern-cmake/chapters/intro/installing.html

Python 

    $ pip install cmake>=3.12   
---
#### Not found 'FFmpeg':

Find FFmppeg lib at:  
> https://www.ffmpeg.org/download.html

**Linux**:

    sudo apt-get install ffmpeg
---
#### Not found 'libjpeg-turbo':

Find FFmppeg lib at: 
> https://libjpeg-turbo.org/Documentation/OfficialBinaries

**Linux**:

    $ sudo apt install libturbojpeg
---
#### Not found Redis or Kafka server:

- Make sure your Redis or Kafka server started.
- Make sure correct username & password.

#### Not found 'Could NOT find OpenSSL':

SSL development libraries have to be installed
> https://github.com/openssl/openssl#build-and-install

Ubuntu:

    $ apt-get install libssl-dev libffi-dev

   
___

Develop by: TinDang   
OS: Ubuntu Bionic 18.4  
Python: 3.6.9 stable
