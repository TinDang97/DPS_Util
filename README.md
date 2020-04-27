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

Ver 1.1.1-1.1.7:
- Fix bugs
- Added UniqueTypeDict
- Fix bugs
- Changed module fixdict -> defaultdict
- Changed FixedDict -> DefaultDict
- Changed FixedTypeDict -> DefaultTypeDict
- Fix bugs of attrdict.AttrDict, attrdict.UniqueTypeDict, attrdict.DefaultDict
- Fix bugs of compress_list, decompress_list

Ver 1.1.8:
- Added Environment

Ver 1.1.9-1.1.18:
- Added Environment.to_lower -> Useful when extracting to kwarg.
- Fix bugs attrdict
- Changed hashfunc default from sha1 -> md5
- Support __call__ of DefaultDict, which update data from buffer.
- Fix bugs Environment

Ver 1.1.19:
- Added pre-define output array of decompress_ndarray
- Support add new key environment.

Ver 1.1.20-1.1.21:
- Edit logger
- Fix bugs vector.pool
- Changed VectorPool base on numpy.mmap -> VectorPoolMMap
- Added VectorPool base on numpy.ndarray
- Restructure VectorPoolBase -> Speed up, fix bugs
- Added VectorPoolBase.insert

Ver 1.1.22:
- Add document of compression, media module.
- Fix bugs image
- Change media.image.scale -> media.image.zoom

Ver 1.1.23-24:
- Fix bugs log
- Fix bugs VectorPoolMMap.MIN_SIZE

Ver 1.1.25-26:
- Fix bugs attrdict
- Added add, remove method attrdict
- Rename variable of pool.VectorPoolBase

Ver 1.1.27:
- Change namespace attrdict.UniqueTypeDict -> attrdict.TypedDict
- Restructure attdict.AttrDict, attdict.DefaultDict, attrdict.TypedDict
- Support annotations alias
- Speed up attrdict
- Support initial attrdict with iterable, generator.
- Support to_buffer method with optimize compression.
- Support recreate dict with from_buffer.
- Lightweight than pickle.dumps

Ver 1.2.0:
- CV:
  - Added Face Aligner
  - Transforms image methods
- media.image:
  - Added scale, crop_center, flip
  - Fix bugs
- media.video: support FPS report.
- triangle (new)


### Todo:
- Attr support constant typing
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

__VectorPoolBase__ handle _numpy.ndarray_ that save a lot of RAM by re-use existed memory space.

Easier than __numpy.ndarray__. Likely __list__.

- Reduce RAM
- High Speed IO
- Auto Scale
- Support backup and recovery data in a second.

#### Media:

Implemented __OpenCV__ with:

- TurboJPEG (https://github.com/lilohuang/PyTurboJPEG)
- FFmpeg(https://github.com/kkroening/ffmpeg-python) 

To: improve read & write (image, video) IO speed.
- Faster than __OpenCV__:
  - 2-6x with JPEG Encoder
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
- **DefaultDict**: help cover your dict with (keys, values) that was defined before. Implement from AttrDict
- **TypedDict**: help cover your dict when set item. Implement from AttrDict.
- **DefaultTypeDict**: dict only access one type for all element.
- Support initial attrdict with iterable, generator.
- Support to_buffer method with optimize compression.
- Support recreate dict with from_buffer.
- Lightweight than pickle.dumps

#### Environement:
- **Environment**: Auto pair environment parameters fit with default, which provided before. 

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
OS: Ubuntu Bionic 18.04  
Python: 3.6.9 stable
