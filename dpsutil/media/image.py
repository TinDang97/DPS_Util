import io
import os

import cv2
import numpy
from turbojpeg import TurboJPEG, TJPF_RGB, TJPF_BGR

from .tool import hex2rgb, color_rgb2bgr, image_info, JPEG_FORMAT, JPEG2000_FORMAT
from .constant import PX_BGR, PX_RGB, DEFAULT_QUALITY, INTER_DEFAULT, ENCODE_PNG, ENCODE_JPEG
from .constant import FLIP_HORIZONTAL, FLIP_VERTICAL, FLIP_BOTH

"""
This tool implement from OpenCV. Can you find more options at https://github.com/opencv/opencv
Drawing: https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

* All method still cover (WIDTH, HEIGHT)
"""

jpeg_compressor = TurboJPEG()


def imencode(img, encode_type=ENCODE_JPEG, quality=DEFAULT_QUALITY, color_format=PX_BGR):
    """
    This function implement from cv2.imencode.
    Ref PNG: https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#imencode
    Ref JPEG: https://github.com/kkroening/ffmpeg-python
    Options:
    - encode_type: ENCODE_JPEG, ENCODE_PNG - Output format of image.
    - quality: set quality of image after encode. 0 -> 100. *Note: 100 - Lossless compression
    - color_format: Image - [B, G, R] or [R, G, B) - Format of color image input.

    Faster 2-6x at JPEG encoder. Otherwise, 1.1x.
    """
    if encode_type == ENCODE_JPEG:
        color_format = TJPF_RGB if color_format == PX_RGB else TJPF_BGR
        buffer = jpeg_compressor.encode(img, quality=quality, pixel_format=color_format)
    else:
        if color_format == PX_RGB:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        quality = max(0, min(int(quality / 10) - 1, 9))
        _, buffer = cv2.imencode(ENCODE_PNG, img, [cv2.IMWRITE_PNG_COMPRESSION, quality])
        buffer = buffer.tobytes()
    return buffer


def imdecode(buffer, color_format=PX_BGR) -> numpy.ndarray:
    """
    This function implement from cv2.imencode.
    Ref: https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#imdecode
    Options:
    - color_format: Image - [B, G, R] or [R, G, B) - Format of color image input.
    """
    ext, (_, _) = image_info(buffer)

    if ext in [JPEG_FORMAT, JPEG2000_FORMAT]:
        color_format = TJPF_RGB if color_format == PX_RGB else TJPF_BGR
        return jpeg_compressor.decode(buffer, pixel_format=color_format)

    image = cv2.imdecode(numpy.frombuffer(buffer, dtype=numpy.uint8), 1)
    if color_format == PX_RGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def imread(img_path, color_format=PX_BGR):
    """
    Read image from file with optimize of speed.
    """
    assert isinstance(img_path, (str, io.BufferedIOBase))

    if type(img_path) is str:
        img_path = open(img_path, 'rb')

    buffer = img_path.read()
    return imdecode(buffer, color_format=color_format)


def imwrite(img, img_path, encode_type=ENCODE_JPEG, quality=95, color_format=PX_BGR, over_write=False):
    """
    Write image into file with best encoder.
    Faster 2-6x at JPEG encoder. Otherwise, 1.1x.

    color_format: current color format of image.
    """
    assert isinstance(img_path, (str, io.BufferedIOBase))

    if isinstance(img_path, str):
        ext = ".jpg"
        if encode_type == ENCODE_PNG:
            ext = ".png"

        if not ext == img_path[-4:]:
            img_path = f"{img_path}{ext}"

        if os.path.isfile(img_path) and not over_write:
            raise FileExistsError

        img_path = open(img_path, 'wb')

    with img_path:
        img_path.write(imencode(img, encode_type=encode_type, quality=quality, color_format=color_format))


def crop(img, box=(0, 0, 0, 0)) -> numpy.ndarray:
    """
    Crop media with margin.

    box = (x, y, width, height). (x, y) are location top-left of box.

    :return: cropped media
    """
    if not numpy.any(box):
        return img

    (max_height, max_width, _) = img.shape
    (x, y, w, h) = box

    x = int(round(max(x, 0)))
    y = int(round(max(y, 0)))
    w = int(round(min(w, max_width)))
    h = int(round(min(h, max_width)))
    return img[y:h, x:w]


def crop_margin(img: numpy.ndarray, margin_size: float, box=(0, 0, 0, 0)) -> numpy.ndarray:
    x, y, w, h = box
    w = w + margin_size
    h = h + margin_size
    x = x - margin_size
    y = y - margin_size
    return crop(img, (x, y, w, h))


# Todo: https://stackoverflow.com/questions/60029431/how-to-pad-an-array-of-images-with-a-given-color-without-a-for-loop
# torchvision.functional.pad
# def pad(img, pad_value, method="constant"):
#     pass


def crop_center(img, crop_size) -> numpy.ndarray:
    """
    Crop center of image with crop_size

    :raise ValueError. if crop_size > image's size
    """
    height, width = img.shape[:2]
    width_crop, height_crop = crop_size

    if width_crop >= width or height_crop >= height:
        raise ValueError(f"crop_size must be smaller than image's size! {width_crop, height_crop} >= {width, height}")

    x = int(round((width - width_crop) / 2.))
    y = int(round((height - height_crop) / 2.))
    return crop(img, (x, y, width_crop, height_crop))


def resize(img, size, keep_ratio=False, inter_method=INTER_DEFAULT):
    """
    This function resize with keep_ratio. Auto downscale or upscale fit with image's height.
    """
    assert isinstance(img, numpy.ndarray)

    new_width, new_height = size
    old_h, old_w, _ = img.shape

    if old_h == new_height and old_w == new_width:
        return img

    if keep_ratio:
        if new_height == old_h:
            return img

        ratio = new_height / old_h
        new_width = old_w * ratio

    new_width = round(new_width)
    new_height = round(new_height)
    return cv2.resize(img, (new_width, new_height), interpolation=inter_method)


def flip(img, flip_mode):
    if flip_mode == FLIP_VERTICAL:
        axis = 0
    elif flip_mode == FLIP_HORIZONTAL:
        axis = 1
    elif flip_mode == FLIP_BOTH:
        axis = (0, 1)
    else:
        raise ValueError(f"Mode '{flip_mode}' isn't supported. Only: FLIP_VERTICAL, FLIP_HORIZONTAL or FLIP_BOTH")
    return numpy.flip(img, axis=axis)


def zoom(img: numpy.ndarray, zoom_level: float, center=None) -> numpy.ndarray:
    """
    Zoom image at position.

    Params:
        zoom_level: integer. Level of zoom. Example: 1x == 1, 1.5x==1.5, 2x == 2...
        center: center of image's zoomed. Default: center old image.
    """
    assert isinstance(img, numpy.ndarray)
    assert type(center) in [type(None), tuple]

    if zoom_level == 1:
        return img

    (h, w) = img.shape[:2]

    if type(center) is tuple:
        assert len(center) == 2
        assert 0 <= center[0] <= w and 0 <= center[1] <= h, "Out of image's length"
    else:
        center = (w // 2, h // 2)

    rotate_matix = cv2.getRotationMatrix2D(center, 0, float(zoom_level))
    return cv2.warpAffine(img, rotate_matix, (w, h))


def scale(img, box=(0, 0, 0, 0), output_size=None):
    if not numpy.any(box):
        box = (0, 0, *img.shape[:2][::-1])

    img = crop(img, box)

    if output_size:
        return resize(img, output_size)
    return img


def rotate_bound(img, angle) -> numpy.ndarray:
    """
    Rote image without crop image.
    """
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    rotate_matix = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = numpy.abs(rotate_matix[0, 0])
    sin = numpy.abs(rotate_matix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rotate_matix[0, 2] += (new_w / 2) - cX
    rotate_matix[1, 2] += (new_h / 2) - cY
    return cv2.warpAffine(img, rotate_matix, (new_w, new_h))


def rotate_crop(img, angle, center=None) -> numpy.ndarray:
    """
    Rotate image and crop part out of size.
    """
    assert type(center) in [type(None), tuple]

    (h, w) = img.shape[:2]

    if type(center) is tuple:
        assert len(center) == 2
        assert 0 <= center[0] <= w and 0 <= center[1] <= h, "Out of image's length"
    else:
        center = (w // 2, h // 2)

    rotate_matix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    return cv2.warpAffine(img, rotate_matix, (h, w))


def draw_text(img, label, position, color=(0, 0, 255), scale_factor=1, thickness=1) -> numpy.ndarray:
    """
    Draw text at position in image.
    - position: top-left of text
    - color: support tuple and hex_color
    :return:
    """
    if not isinstance(position, tuple):
        position = tuple(position)

    if isinstance(color, str):
        color = hex2rgb(color)
        color = color_rgb2bgr(color)

    return cv2.putText(img, label, position,
                       fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale_factor, color=color, thickness=thickness)


def draw_square(img, position, color=(0, 255, 0)) -> numpy.ndarray:
    """
    Draw text at position in image.
    - position: top-left, bottom-right of square
    - color: support tuple and hex_color
    :return:
    """
    if not isinstance(position, tuple):
        position = tuple(position)

    if isinstance(color, str):
        color = hex2rgb(color)
        color = color_rgb2bgr(color)

    return cv2.rectangle(img, position[0:2], position[2:4], color, 2)


# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result


__all__ = ['imencode', 'imdecode', 'imread', 'imwrite', 'crop', 'resize', 'zoom', 'rotate_bound', 'rotate_crop',
           'draw_text', 'draw_square', 'automatic_brightness_and_contrast', 'ENCODE_PNG', 'ENCODE_JPEG', 'PX_BGR',
           'PX_RGB', 'scale', 'flip', 'crop_margin', 'crop_center']
