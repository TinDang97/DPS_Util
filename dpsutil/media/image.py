import io
import os

import cv2
import numpy
from turbojpeg import TurboJPEG, TJPF_RGB, TJPF_BGR

from .tool import hex2rgb, color_rgb2bgr, image_info, PNG_FORMAT, JPEG_FORMAT, JPEG2000_FORMAT
from .constant import PX_BGR, PX_RGB, DEFAULT_QUALITY, INTER_DEFAULT, ENCODE_PNG, ENCODE_JPEG

"""
This tool implement from OpenCV. Can you find more options at https://github.com/opencv/opencv
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
    :return: buffer
    :rtype: bytes
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


def imdecode(buffer, color_format=PX_BGR):
    """
    This function implement from cv2.imencode.
    Ref: https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#imdecode
    Options:
    - color_format: Image - [B, G, R] or [R, G, B) - Format of color image input.
    :rtype: numpy.ndarray
    """
    ext, _, _ = image_info(buffer)

    if ext in [JPEG_FORMAT, JPEG2000_FORMAT]:
        color_format = TJPF_RGB if color_format == PX_RGB else TJPF_BGR
        return jpeg_compressor.decode(buffer, pixel_format=color_format)

    image = cv2.imdecode(numpy.frombuffer(buffer, dtype=numpy.uint8), 1)
    if color_format == PX_RGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def imread(img_path, color_format=PX_BGR):
    assert isinstance(img_path, (str, io.BufferedIOBase))

    try:
        if type(img_path) is str:
            img_path = open(img_path, 'rb')

        buffer = img_path.read()
        return imdecode(buffer, color_format=color_format)
    except Exception as e:
        raise e
    finally:
        img_path.close()


def imwrite(img, img_path, encode_type=ENCODE_JPEG, quality=95, color_format=PX_BGR, over_write=False):
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


def crop_image(img, box=(0, 0, 0, 0), margin_size=0):
    """
    Crop media with margin.

    box = (x, y, width, height). (x, y) are location top-left of box.

    :return: cropped media
    """
    if not img.any():
        raise Exception("Image is not existed!")

    (max_height, max_width, _) = img.shape
    (x, y, w, h) = box

    x_end = x + w + margin_size
    y_end = y + h + margin_size
    x = x - margin_size
    y = y - margin_size

    x = round(max(x, 0))
    y = round(max(y, 0))
    x_end = round(min(x_end, max_width))
    x_end = round(min(x_end, max_width))

    return img[y:y_end, x:x_end]


def resize(img, size, keep_ratio=False, inter_method=INTER_DEFAULT):
    """
    This function resize with keep_ratio
    :return:
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


def scale(img, ratio):
    assert isinstance(img, numpy.ndarray)

    if ratio == 1:
        return img

    (h, w) = img.shape[:2]
    rotate_matix = cv2.getRotationMatrix2D((w // 2, h // 2), 0, float(ratio))
    return cv2.warpAffine(img, rotate_matix, (h, w))


def rotate_bound(img, angle):
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


def rotate_crop(img, angle, center=None):
    (h, w) = img.shape[:2]

    if not center:
        center = (w // 2, h // 2)

    rotate_matix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    return cv2.warpAffine(img, rotate_matix, (h, w))


def draw_text(img, label, position, color=(0, 0, 255), scale_factor=1, thickness=1):
    """
    Draw text at position in image.
    - position: top-left of text
    - color: support tuple and hex_color
    :return:
    """
    if isinstance(position, numpy.ndarray):
        position = position.tolist()

    if isinstance(position, list):
        position = tuple(position)

    if isinstance(color, str):
        color = hex2rgb(color)
        color = color_rgb2bgr(color)

    return cv2.putText(img, label, position,
                       fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale_factor, color=color, thickness=thickness)


def draw_square(img, position, color=(0, 255, 0)):
    """
    Draw text at position in image.
    - position: top-left, bottom-right of square
    - color: support tuple and hex_color
    :return:
    """
    if isinstance(position, numpy.ndarray):
        position = position.tolist()

    if isinstance(position, list):
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


__all__ = ['imencode', 'imdecode', 'imread', 'imwrite', 'crop_image', 'resize', 'scale', 'rotate_bound', 'rotate_crop',
           'draw_text', 'draw_square', 'automatic_brightness_and_contrast', 'ENCODE_PNG', 'ENCODE_JPEG', 'PX_BGR',
           'PX_RGB']
