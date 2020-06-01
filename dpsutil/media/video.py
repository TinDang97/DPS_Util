import os
import time
from queue import Queue, Empty
from threading import Thread

import ffmpeg
import numpy

from .constant import PX_RGB, PX_BGR, DEFAULT_QUALITY, THUMBNAIL_RESOLUTION, SD_RESOLUTION
from .image import ENCODE_JPEG, ENCODE_PNG, imwrite
from .image import imencode, imdecode
from .tool import show_image, destroy_windows
from ..multiprocess.threading import QueueOverflow

# Options pixel format. `ffmpeg -pix_fmts` get more options
RGB24 = "rgb24"
BGR24 = "bgr24"
YUV420P = "yuv420p"
YUV422P = "yuv422p"
YUV444P = "yuv444p"

# Option encode format. `ffmpeg -codecs` get more options
H264_ENCODER = "libx264"
H265_ENCODER = "libx265"

# LOGGING
# Show nothing at all; be silent.
LOG_QUITE = "quite"

# Only show fatal errors which could lead the process to crash, such as an assertion failure.
# This is not currently used for anything.
LOG_ERROR = "error"

# Show all warnings and errors. Any message related to possibly incorrect or unexpected events will be shown.
LOG_WARNING = "warning"

# Show informative messages during processing. This is in addition to warnings and errors.
LOG_INFO = "info"

# Show everything, including debugging information.
LOG_DEBUG = "debug"


class CaptureError(Exception):
    """
    CaptureError contain all error of VideoCapture

    Error message is error detail if msg is ffmpeg._run.Error

    Parameters
    ----------
    msg: str | Exception | ffmpeg.Error
        Error message
    """

    def __init__(self, msg):
        if isinstance(msg, ffmpeg.Error) and hasattr(msg, 'stderr'):
            reason = msg.stderr.decode().strip().split("\n")[-1]
            msg = f"{msg.__repr__()}\noutput: {msg.stdout}\nreason: {reason}"
        super(Exception, self).__init__(msg)


class Frame(object):
    def __init__(self, buffer, size):
        self.size = size
        self.buffer = buffer

    def __bool__(self):
        return True

    def __bytes__(self):
        return self.buffer

    def tobytes(self):
        return self.buffer

    def decode(self):
        return numpy.frombuffer(self.buffer, dtype=numpy.uint8).reshape((self.size[1], self.size[0], 3))

    def encode(self, encode_type=ENCODE_JPEG, quality=DEFAULT_QUALITY):
        return imencode(self.decode(), encode_type=encode_type, quality=quality)

    def save(self, img_path, encode_type=ENCODE_JPEG, quality=DEFAULT_QUALITY, over_write=False):
        if os.path.isfile(img_path) and not over_write:
            raise FileExistsError

        with open(img_path, "wb") as img:
            img.write(self.encode(encode_type=encode_type, quality=quality))


class VideoIterator(object):
    """
    VideoCapture iterator

    Parameters
    ----------
    stream: ffmpeg.OutputStream

    """
    __STOP_BYTES = b'U0U='

    def __init__(self, stream, size, cache_frames=30, is_stream=False):
        # prepare reader
        assert isinstance(stream, ffmpeg.nodes.OutputStream)
        self.stream = ffmpeg.run_async(stream, pipe_stdout=True)
        self.size = size

        # prepare pool frame
        self.pool_frames = QueueOverflow(cache_frames) if is_stream else Queue(cache_frames)
        self.thread = Thread(target=self.__read_buffer)
        self.read_byte_size = self.size[0] * self.size[1] * 3

        # metadata
        self.counter = 0
        self.start_time = 0
        self.end_time = 0

    def __iter__(self):
        if not self.thread.is_alive():
            self.start()
        return self

    def __next__(self):
        self.counter += 1
        frame = self.get_frame()

        if not frame:
            raise StopIteration
        return frame

    def fps(self):
        end_time = self.end_time
        if self.thread.is_alive():
            end_time = time.time()
        return int(round(self.counter // (end_time - self.end_time)))

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()

        self.start_time = time.time()
        self.end_time = 0
        self.counter = 0

    def get_frame(self, time_out=10):
        if self.end_time > 0:
            return None

        try:
            buffer = self.pool_frames.get(timeout=time_out)

            if buffer == self.__STOP_BYTES:
                return None
        except Empty:
            return None

        return Frame(buffer, self.size)

    def __read_buffer(self):
        while self.stream.poll() is None:
            buffer = self.stream.stdout.read(self.read_byte_size)

            if not buffer:
                continue

            self.pool_frames.put(buffer)

        # Stop record
        self.end_time = time.time()
        self.pool_frames.put(self.__STOP_BYTES)

    def close(self):
        self.stream.terminate()

    __enter__ = (lambda self: self)
    __exit__ = (lambda self, exc_type, exc_val, exc_tb: self.close())


class VideoInfo(object):
    def __init__(self, src):
        # check source
        super().__init__()

        try:
            info_streams = ffmpeg.probe(src)
            self.info = next(stream for stream in info_streams['streams'] if stream['codec_type'] == "video")
        except ffmpeg.Error as e:
            raise CaptureError(e) from None
        except StopIteration:
            raise CaptureError("No video stream from source!") from None

        # meta-data
        self.src = src
        self.fps = round(eval(self.info['r_frame_rate']))
        self.height = self.info['height']
        self.width = self.info['width']


class VideoCapture(object):
    """
    VideoCapture implement FFmpeg - High Performance Video IO.

    Parameters
    ----------
    src: str | int
        Support local camera device (int), RTSP, Video.

    Raises
    ------
    CaptureError
        error during initial capture

    Examples
    --------
    Create capture and show using OpenCV.

    >>> capture = VideoCapture(0)
    capture camera 0

    # capture = VideoCapture("rtsp://****")
    # capture = VideoCapture("file_path")

    >>> with capture.read(realtime=True) as reader:
    >>>    for frame in reader:
    >>>        frame_ndarray = frame.decode()
    >>>        if not show_image(, 'asd', windows_size=FHD_RESOLUTION):
    >>>            break

    OR

    >>> reader = capture.read(realtime=True)
    >>> for frame in reader:
    >>>     frame_ndarray = frame.decode()
    >>>     if not show_image(, 'asd', windows_size=FHD_RESOLUTION):
    >>>         break
    >>> reader.close()

    """

    def __init__(self, src):
        self.is_stream = False

        # options
        opts = {
            'an': None
        }

        # camera
        if isinstance(src, int):
            src = f"/dev/video{src}"
            self.is_stream = True

        # stream
        if src.startswith("rtsp"):
            opts["rtsp_transport"] = "tcp"
            opts["re"] = None
            self.is_stream = True

        # source metadata
        self.__meta = VideoInfo(src)

        # create input cmd
        self.__input_stream = ffmpeg.input(src, **opts)

    def __repr__(self):
        return f"VideoCapture implement FFmpeg - High Performance Video IO.\n" \
               f"Source: {self.source}\n" \
               f"Size: {self.size}\n" \
               f"FPS: {self.fps}"

    @property
    def source(self):
        return self.__meta.src

    @property
    def size(self):
        return self.__meta.width, self.__meta.height

    @property
    def fps(self):
        return self.__meta.fps

    def read(self, output_size=None, keep_ratio=True, duration=0, pix_fmt=RGB24, log_level=LOG_ERROR):
        """
        Generate VideoIterator which yield once frame by frame.

        Parameters
        ---------
        output_size: tuple[int, int]
            Output size of stream

        keep_ratio: bool
            If True, width will change to fix with height ratio. w *= h_new / h_old

        duration: int
            Limited stream duration if set.

        pix_fmt: str
            (Default: RGB24) Format of each pixel in frame.

        log_level: LogLevel
            Log level of ffmpeg

        Returns
        -------
        VideoIterator
            Frame fetcher.

        Raises
        ------
        CaptureError
            If output_size <= (-1, -1)
        """
        input_stream = self.__input_stream

        if not output_size:
            output_size = self.size

        if keep_ratio:
            output_size = (
                int(round(self.__meta.width * (output_size[1] / self.__meta.height))),
                output_size[1]
            )

        output_options = {
            "format": 'rawvideo',
            "pix_fmt": pix_fmt,
            "loglevel": log_level,
            "s": f'{output_size[0]}x{output_size[1]}'
        }

        if duration:
            output_options['t'] = duration

        capture = ffmpeg.output(
            input_stream,
            'pipe:',
            **output_options
        )

        return VideoIterator(
            capture,
            self.size,
            is_stream=self.is_stream
        )

    def write_images(self, folder_path, prefix="", output_size=None, keep_ratio=True, duration=0, pix_fmt=RGB24,
                     encode_type=ENCODE_JPEG, quality=DEFAULT_QUALITY, over_write=False, log_level=LOG_ERROR):
        """
        Read all frame from source and write it to images.

        Same manual way:
            Get reader with capture.read and write frame into file.

        Parameters
        ----------
        folder_path: str
            Path of output folder

        prefix: str
            prefix of each image's name

        output_size: tuple[int, int]
            Output size of stream

        keep_ratio: bool
            If True, width will change to fix with height ratio. w *= h_new / h_old

        duration: int
            Limited stream duration if set.

        pix_fmt: str
            (Default: RGB24) Format of each pixel in frame.

        encode_type: int
            ENCODE_JPEG (default) | ENCODE_PNG. Output encode format of images

        quality: int
            0 -> (default) 95 -> 100. Quality of images

        over_write: bool
            force write data in existed file.

        log_level: LogLevel
            Log level of ffmpeg
        """
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        ext = ".jpg"
        if encode_type == ENCODE_PNG:
            ext = ".png"

        reader = self.read(output_size=output_size, keep_ratio=keep_ratio, duration=duration, pix_fmt=pix_fmt,
                           log_level=log_level)

        for idx, frame in enumerate(reader):
            file_path = f"{os.path.abspath(folder_path)}/{prefix}{idx}{ext}"
            imwrite(frame.decode(), file_path, encode_type=encode_type, quality=quality, over_write=over_write)

        reader.close()

    def write(self, file_path, over_write=False, encoder=H265_ENCODER, pix_fmt=YUV420P,
              output_size=None, keep_ratio=True, fps=-1, duration=0,
              preview=False, preview_size=SD_RESOLUTION,
              log_level=LOG_ERROR):
        """
        Capture input stream to file. With preview options

        Preview mode can down speed processing 1.5x -> 2x.

        Parameters
        ----------
        file_path: str
            Path of output file

        over_write: bool
            Overwrite existed file if True.

        encoder: str
            (Default: H265_ENCODER) Encode codec.

        pix_fmt: str
            (Default: YUV420P) Pixel format

        output_size: tuple[int, int]
            Output size

        output_size: tuple[int, int]
            Output size of stream

        keep_ratio: bool
            If True, width will change to fix with height ratio. w *= h_new / h_old

        fps: int
            (Default: -1 - autoset) FPS of output video

        duration: int
            (Default: 0 - infinite) Limited record time if set.

        preview: bool
            Show frame during process. If True.

        preview_size: tuple[int, int]
            Window previewer size.

        log_level: LogLevel
            Log level of ffmpeg
        """

        if os.path.isfile(file_path) and not over_write:
            raise FileExistsError

        if not output_size:
            output_size = self.size

        if keep_ratio:
            output_size = (
                int(round(self.__meta.width * (output_size[1] / self.__meta.height))),
                THUMBNAIL_RESOLUTION[1]
            )

        # file output settings
        output_options = {
            "pix_fmt": pix_fmt,
            "loglevel": log_level,
            "c:v": encoder,
            "crf": 27,
            'preset': 'veryfast',
            's': f'{output_size[0]}x{output_size[1]}'
        }

        if duration:
            output_options['t'] = duration

        if encoder == H265_ENCODER:
            output_options["x265-params"] = f"log-level={log_level if log_level != 'quiet' else -1}"

        # handle FPS
        if fps > 0:
            # manual set
            output_options["r"] = fps
        elif self.is_stream:
            # sync with time
            output_options["vsync"] = "vfr"
        else:
            # same source file
            output_options["r"] = self.fps

        capture_output = self.__input_stream \
            .output(file_path, **output_options) \
            .overwrite_output()

        # pipe output settings if preview.
        if preview:
            preview_output_size = THUMBNAIL_RESOLUTION

            if keep_ratio:
                preview_output_size = (
                    int(round(self.__meta.width * (preview_output_size[1] / self.__meta.height))),
                    preview_output_size[1]
                )

            pipe_output_opts = {
                "format": "rawvideo",
                "pix_fmt": RGB24,
                "loglevel": log_level,
                's': f'{preview_output_size[0]}x{preview_output_size[1]}'
            }

            if duration:
                pipe_output_opts['t'] = duration

            pipe_output = self.__input_stream.output('pipe:', **pipe_output_opts)
            capture_output = ffmpeg.merge_outputs(pipe_output, capture_output)

            capture_output = VideoIterator(
                capture_output,
                preview_output_size,
                is_stream=self.is_stream
            )
            window_name = f"Preview - {self.__meta.src}"

            for frame in capture_output:
                if not show_image(frame.decode(), window_name, windows_size=preview_size):
                    break

            destroy_windows(window_name)
            capture_output.close()
        else:
            capture_output.run()


def images2video(file_path, images_list, fps,
                 pix_fmt=YUV420P, encoder=H265_ENCODER,
                 output_size=None, keep_ratio=True,
                 over_write=False, duration=0, log_level=LOG_ERROR):
    """
    Make video file from sequence images.

    *images_list: list path file of images.

    Tip: `fast way get list of images`.
    1) Get all file path by glob.glob
    2) Sort with dpsutil.sort.natsorted

    Parameters
    ----------
    file_path: str
        Path of output video

    images_list: list[str]
        List of images which used to re-sequence to video

    fps: int
        FPS of output video

    encoder: str
            (Default: H265_ENCODER) Encode codec.

    pix_fmt: str
        (Default: YUV420P) Pixel format

    output_size: tuple[int, int]
        Output size of output video

    keep_ratio: bool
        If True, width will change to fix with height ratio. w *= h_new / h_old

    over_write: bool
        Overwrite existed file if True.

    duration: int
        (Default: 0 - infinite) Limited record time if set.

    log_level: LogLevel
            Log level of ffmpeg
    """
    if os.path.isfile(file_path) and not over_write:
        raise FileExistsError

    img_info = ffmpeg.probe(images_list[0])['streams'][0]
    origin_size = (img_info['width'], img_info['height'])

    if not output_size:
        output_size = origin_size

    if keep_ratio:
        output_size = (
            int(round(origin_size[0] * (output_size[1] / origin_size[1]))),
            output_size[1]
        )

    output_options = {
        "pix_fmt": pix_fmt,
        "c:v": H265_ENCODER,
        'loglevel': log_level.value
    }

    if encoder == H265_ENCODER:
        output_options["x265-params"] = f"log-level={log_level if log_level != 'quiet' else -1}"

    if duration:
        output_options['t'] = duration

    capture_output = ffmpeg.input("pipe:", format='rawvideo', pix_fmt=RGB24, framerate=fps,
                                  s=f'{output_size[0]}x{output_size[1]}') \
        .output(file_path, **output_options) \
        .overwrite_output() \
        .run_async(pipe_stdin=True)

    try:
        for image_path in images_list:
            with open(image_path, "rb") as image_file:
                img = imdecode(image_file.read())
                capture_output.stdin.write(img.tobytes())
    except Exception as e:
        raise e
    finally:
        capture_output.stdin.close()
        capture_output.wait()


__all__ = ['images2video', 'VideoCapture', 'PX_RGB', 'PX_BGR', 'ENCODE_JPEG', 'ENCODE_PNG']
