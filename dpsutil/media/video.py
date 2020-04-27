import os
import time
from collections import deque
from queue import Queue
from threading import Thread, Lock, Condition

import ffmpeg
import numpy

from .constant import PX_RGB, PX_BGR, DEFAULT_QUALITY
from .image import ENCODE_JPEG, ENCODE_PNG, imwrite
from .image import imencode, imdecode
from .tool import show_image

FPS_DEFAULT = 0
SIZE_DEFAULT = (-1, -1)

PIXEL_RGB_FORMAT = "rgb24"
PIXEL_BGR_FORMAT = "bgr24"


class Frame(object):
    def __init__(self, buffer, size, pixel_format=PIXEL_RGB_FORMAT):
        assert pixel_format in [PIXEL_RGB_FORMAT, PIXEL_BGR_FORMAT]

        self.size = size
        self.buffer = buffer

        if pixel_format == PIXEL_RGB_FORMAT:
            self.pix_fmt = PX_RGB
        else:
            self.pix_fmt = PX_BGR

    def __bytes__(self):
        return self.buffer

    def tobytes(self):
        return self.buffer

    def decode(self):
        return numpy.frombuffer(self.buffer, dtype=numpy.uint8).reshape((*self.size, 3))

    def encode(self, encode_type=ENCODE_JPEG, quality=DEFAULT_QUALITY):
        return imencode(self.decode(), encode_type=encode_type, quality=quality, color_format=self.pix_fmt)

    def save(self, img_path, encode_type=ENCODE_JPEG, quality=DEFAULT_QUALITY, over_write=False):
        if os.path.isfile(img_path) and not over_write:
            raise FileExistsError

        with open(img_path, "wb") as img:
            img.write(self.encode(encode_type=encode_type, quality=quality))


class VideoIterator(object):
    class Box:
        def __init__(self):
            self.box = deque(maxlen=1)
            self.locker = Lock()
            self.put_lock = Condition(self.locker)
            self.get_lock = Condition(self.locker)

        def put(self, data):
            with self.put_lock:
                self.box.append(data)
                self.get_lock.notify()

        def get(self, timeout=30):
            with self.get_lock:
                end_time = time.time() + timeout
                while not len(self.box):
                    remaining = end_time - time.time()
                    if remaining <= 0.0:
                        raise TimeoutError
                    self.get_lock.wait(remaining)

                data = self.box.popleft()
                self.put_lock.notify()
                return data

    def __init__(self, stream, size, pixel_format=PIXEL_RGB_FORMAT, get_latest=False, cache_frames=30):
        self.stream = ffmpeg.run_async(stream, pipe_stdout=True)
        self.size = size
        self.pix_fmt = pixel_format

        self.pool_frames = self.Box() if get_latest else Queue(maxsize=cache_frames)
        self.thread = Thread(target=self.read_buffer)
        self.read_byte_size = self.size[0] * self.size[1] * 3
        self.counter = 0
        self.start_time = 0

    def __iter__(self):
        if not self.thread.is_alive():
            self.start()
        return self

    def __next__(self):
        self.counter += 1
        return self.get_frame()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()
        self.start_time = time.time()
        self.counter = 0

    def close(self):
        self.stream.terminate()
        return self.counter/(time.time() - self.start_time)

    def get_frame(self, time_out=30):
        buffer = self.pool_frames.get(timeout=time_out)
        return Frame(buffer, self.size, pixel_format=self.pix_fmt)

    def read_buffer(self):
        while self.stream.poll() is None:
            buffer = self.stream.stdout.read(self.read_byte_size)

            if not buffer:
                continue

            self.pool_frames.put(buffer)


class VideoCapture(object):
    def __init__(self, src, fps=FPS_DEFAULT, output_size=SIZE_DEFAULT, keep_ratio=True, pixel_format=PX_BGR):
        """
        VideoCapture implement FFmpeg - High Performance Video IO.
        - src: support local camera device, RTSP - TCP, Video & Image.
                Example:
                     VideoCapture(0) - capture camera 0
                     VideoCapture("rtsp://****")
                     VideoCapture("file_path")
        - fps: same source's fps, If not set.
        - output_size: same source's size, If not set.
        - keep_ratio: keep ratio of output_size from source's size, If output_size has been set.
        - pixel_format: support PX_RGB, PX_BGR

        Example:
            ```
            capture = VideoCapture(0)

            with capture.read(realtime=True) as reader:
                for frame in reader:
                    frame_ndarray = frame.decode()
                    if not show_image(, 'asd', windows_size=FHD_RESOLUTION):
                        break

            OR

            reader = capture.read(realtime=True)
            for frame in reader:
                frame_ndarray = frame.decode()
                if not show_image(, 'asd', windows_size=FHD_RESOLUTION):
                    break
            reader.close()
            ```
        :return
        """
        if isinstance(src, int):
            src = f"/dev/video{src}"

        opts = {
            'an': None
        }
        if "rtsp" in src:
            opts["rtsp_transport"] = "tcp"
            opts["re"] = None

        self.__src = src
        self.__info = ffmpeg.probe(src)['streams'][0]
        self.__capture = ffmpeg.input(src, **opts)

        if fps > 0:
            self.__capture = ffmpeg.filter(self.__capture, "fps", fps=fps, round="up")
            self.fps = fps
        else:
            self.fps = round(eval(self.__info['r_frame_rate']))

        if output_size > SIZE_DEFAULT:
            w, h = output_size

            if keep_ratio:
                w = -1

            self.__capture = ffmpeg.filter(self.__capture, "scale", w=w, h=h)
            self.__height = h
            self.__width = round(self.__info['width'] * (h / self.__info['height']))
        else:
            self.__height = self.__info['height']
            self.__width = self.__info['width']

        self.pix_fmt = PIXEL_RGB_FORMAT if pixel_format == PX_RGB else PIXEL_BGR_FORMAT

    def __repr__(self):
        return f"VideoCapture implement FFmpeg - High Performance Video IO.\n" \
               f"Source: {self.source}\n" \
               f"Size: {self.size}\n" \
               f"FPS: {self.fps}"

    @property
    def source(self):
        return self.__src

    @property
    def size(self):
        return self.__width, self.__height

    def write_images(self, folder_path, prefix="", encode_type=ENCODE_JPEG, quality=DEFAULT_QUALITY, over_write=False):
        """
        Read all frame from source and write it to images.
        - folder_path: output folder
        - prefix: prefix of each image's name
        - encode_type: ENCODE_JPEG, ENCODE_PNG
        - quality: 0 -> 100
        - over_write: force write data in existed file.

        More info -> dpsutil.media.image.imwrite
        :return:
        """
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        ext = ".jpg"
        if encode_type == ENCODE_PNG:
            ext = ".png"

        with self.read() as capture:
            for idx, frame in enumerate(capture):
                file_path = f"{os.path.abspath(folder_path)}/{prefix}{idx}{ext}"
                imwrite(frame.decode(), file_path, encode_type=encode_type, quality=quality, color_format=self.pix_fmt,
                        over_write=over_write)

    def write(self, file_path, over_write=False, realtime=False, preview=True, output_pixel_fm="yuv420p", timeout=30):
        """
        Get more output_pixel_fm in terminal: `ffmpeg -pix_fmts`
        :return:
        """

        if os.path.isfile(file_path) and not over_write:
            raise FileExistsError

        capture_output = ffmpeg \
            .input('pipe:', format='rawvideo', s=f'{self.__width}x{self.__height}', pix_fmt=self.pix_fmt) \
            .output(file_path, pix_fmt=output_pixel_fm, r=self.fps, crf=18) \
            .overwrite_output() \
            .run_async(pipe_stdin=True)

        capture_input = self.read(realtime=realtime)
        for frame in capture_input:
            capture_output.stdin.write(frame.buffer)
            if preview:
                if not show_image(frame.decode(), f"Preview - {self.__src}"):
                    break

        capture_input.close()
        capture_output.stdin.close()
        capture_output.wait(timeout=timeout)

    def read(self, realtime=False, cache_frames=30):
        """
        Generate VideoIterator which yield once frame at time.
        - realtime: If True, returned frame which always is the latest frame.
        - cache_frames: Useful when realtime is False, which set number of cached frame in queue
        :return:
        """
        capture = ffmpeg.output(self.__capture, 'pipe:', format='rawvideo', pix_fmt=self.pix_fmt, loglevel="quiet")
        return VideoIterator(capture, (self.__height, self.__width), pixel_format=self.pix_fmt, get_latest=realtime,
                             cache_frames=cache_frames)


def images2video(file_path, images_list, fps, output_size=(0, 0), over_write=False):
    """
    Make video file from sequence images.

    *images_list: list path file of images.

    Tip: `fast way get list of images`.
    1) Get all file path by glob.glob
    2) Sort with dpsutil.sort.natsorted
    """
    if os.path.isfile(file_path) and not over_write:
        raise FileExistsError

    if output_size <= (0, 0):
        img_info = ffmpeg.probe(images_list[0])['streams'][0]
        output_size = (img_info['width'], img_info['height'])

    capture_output = ffmpeg.input("pipe:", format='rawvideo', pix_fmt=PIXEL_BGR_FORMAT, framerate=fps,
                                  s=f'{output_size[0]}x{output_size[1]}') \
        .output(file_path, pix_fmt="yuv420p", **{"c:v": "libx264"}) \
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
