import io
import re
import struct

import cv2
from .constant import SD_RESOLUTION

_UNIT_KM = -3
_UNIT_100M = -2
_UNIT_10M = -1
_UNIT_1M = 0
_UNIT_10CM = 1
_UNIT_CM = 2
_UNIT_MM = 3
_UNIT_0_1MM = 4
_UNIT_0_01MM = 5
_UNIT_UM = 6
_UNIT_INCH = 6

_TIFF_TYPE_SIZES = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
}

PNG_FORMAT = "png"
JPEG_FORMAT = "jpg"
JPEG2000_FORMAT = "j2"
TIFF_FORMAT = "tiff"
GIF_FORMAT = "gif"


def _convertToDPI(density, unit):
    if unit == _UNIT_KM:
        return int(density * 0.0000254 + 0.5)
    elif unit == _UNIT_100M:
        return int(density * 0.000254 + 0.5)
    elif unit == _UNIT_10M:
        return int(density * 0.00254 + 0.5)
    elif unit == _UNIT_1M:
        return int(density * 0.0254 + 0.5)
    elif unit == _UNIT_10CM:
        return int(density * 0.254 + 0.5)
    elif unit == _UNIT_CM:
        return int(density * 2.54 + 0.5)
    elif unit == _UNIT_MM:
        return int(density * 25.4 + 0.5)
    elif unit == _UNIT_0_1MM:
        return density * 254
    elif unit == _UNIT_0_01MM:
        return density * 2540
    elif unit == _UNIT_UM:
        return density * 25400
    return density


def _convertToPx(value):
    matched = re.match(r"(\d+)(?:\.\d)?([a-z]*)$", value)
    if not matched:
        raise ValueError("unknown length value: %s" % value)
    else:
        length, unit = matched.groups()
        if unit == "":
            return int(length)
        elif unit == "cm":
            return int(length) * 96 / 2.54
        elif unit == "mm":
            return int(length) * 96 / 2.54 / 10
        elif unit == "in":
            return int(length) * 96
        elif unit == "pc":
            return int(length) * 96 / 6
        elif unit == "pt":
            return int(length) * 96 / 6
        elif unit == "px":
            return int(length)
        else:
            raise ValueError("unknown unit type: %s" % unit)


def image_info(src):
    """
    Implement: https://github.com/shibukawa/imagesize_py
    Return raw format, (width, height) for a given img file content
    no requirements
    :rtype str, (int, int)
    """
    assert isinstance(src, (bytes, io.BufferedReader, str))
    height = -1
    width = -1
    cursor = 0
    image_format = None

    if type(src) is str:
        src = open(src, 'rb')

    if type(src) is io.BufferedReader:
        buffer = src.read()
        src.close()
    else:
        buffer = src

    head = buffer[:24]
    size = len(head)
    # handle GIFs
    if size >= 10 and head[:6] in (b'GIF87a', b'GIF89a'):
        # Check to see if content_type is correct
        try:
            width, height = struct.unpack("<hh", head[6:10])
            image_format = GIF_FORMAT
        except struct.error:
            raise ValueError("Invalid GIF file")
    # see png edition spec bytes are below chunk length then and finally the
    elif size >= 24 and head.startswith(b'\211PNG\r\n\032\n') and head[12:16] == b'IHDR':
        try:
            width, height = struct.unpack(">LL", head[16:24])
            image_format = PNG_FORMAT
        except struct.error:
            raise ValueError("Invalid PNG file")
    # Maybe this is for an older PNG version.
    elif size >= 16 and head.startswith(b'\211PNG\r\n\032\n'):
        # Check to see if we have the right content type
        try:
            width, height = struct.unpack(">LL", head[8:16])
            image_format = PNG_FORMAT
        except struct.error:
            raise ValueError("Invalid PNG file")
    # handle JPEGs
    elif size >= 2 and head.startswith(b'\377\330'):
        try:
            size = 2
            ftype = 0
            while not 0xc0 <= ftype <= 0xcf or ftype in [0xc4, 0xc8, 0xcc]:
                cursor += size
                byte = buffer[cursor:cursor + 1]
                cursor += 1
                while ord(byte) == 0xff:
                    byte = buffer[cursor:cursor + 1]
                    cursor += 1
                ftype = ord(byte)
                size = struct.unpack('>H', buffer[cursor:cursor + 2])[0] - 2
                cursor += 2
            # We are at a SOFn block
            cursor += 1  # Skip `precision' byte.
            height, width = struct.unpack('>HH', buffer[cursor:cursor + 4])
            image_format = JPEG_FORMAT
        except struct.error:
            raise ValueError("Invalid JPEG file")
    # handle JPEG2000s
    elif size >= 12 and head.startswith(b'\x00\x00\x00\x0cjP  \r\n\x87\n'):
        cursor = 48
        try:
            height, width = struct.unpack('>LL', buffer[cursor:cursor + 8])
            image_format = JPEG2000_FORMAT
        except struct.error:
            raise ValueError("Invalid JPEG2000 file")
    # handle big endian TIFF
    elif size >= 8 and head.startswith(b"\x4d\x4d\x00\x2a"):
        offset = struct.unpack('>L', head[4:8])[0]
        cursor = offset
        ifdsize = struct.unpack(">H", buffer[cursor:cursor + 2])[0]
        cursor += 2
        for i in range(ifdsize):
            tag, datatype, count, data = struct.unpack(">HHLL", buffer[cursor:cursor + 12])
            if tag == 256:
                if datatype == 3:
                    width = int(data / 65536)
                elif datatype == 4:
                    width = data
                else:
                    raise ValueError("Invalid TIFF file: width column data type should be SHORT/LONG.")
            elif tag == 257:
                if datatype == 3:
                    height = int(data / 65536)
                elif datatype == 4:
                    height = data
                else:
                    raise ValueError("Invalid TIFF file: height column data type should be SHORT/LONG.")
            if width != -1 and height != -1:
                break
        if width == -1 or height == -1:
            raise ValueError("Invalid TIFF file: width and/or height IDS entries are missing.")
        image_format = TIFF_FORMAT
    elif size >= 8 and head.startswith(b"\x49\x49\x2a\x00"):
        offset = struct.unpack('<L', head[4:8])[0]
        cursor = offset
        ifdsize = struct.unpack("<H", buffer[cursor:cursor + 2])[0]
        cursor += 2
        for i in range(ifdsize):
            tag, datatype, count, data = struct.unpack("<HHLL", buffer[cursor:cursor + 12])
            if tag == 256:
                width = data
            elif tag == 257:
                height = data
            if width != -1 and height != -1:
                break
        if width == -1 or height == -1:
            raise ValueError("Invalid TIFF file: width and/or height IDS entries are missing.")
        image_format = TIFF_FORMAT
    return image_format, (width, height)


def image_dpi(filepath):
    """
    Return (x DPI, y DPI) for a given img file content
    no requirements
    :type filepath: Union[str, pathlib.Path]
    :rtype Tuple[int, int]
    """
    xDPI = -1
    yDPI = -1
    with open(str(filepath), 'rb') as fhandle:
        head = fhandle.read(24)
        size = len(head)
        # handle GIFs
        # GIFs doesn't have density
        if size >= 10 and head[:6] in (b'GIF87a', b'GIF89a'):
            pass
        # see png edition spec bytes are below chunk length then and finally the
        elif size >= 24 and head.startswith(b'\211PNG\r\n\032\n'):
            chunkOffset = 8
            chunk = head[8:]
            while True:
                chunkType = chunk[4:8]
                if chunkType == b'pHYs':
                    try:
                        xDensity, yDensity, unit = struct.unpack(">LLB", chunk[8:])
                    except struct.error:
                        raise ValueError("Invalid PNG file")
                    if unit:
                        xDPI = _convertToDPI(xDensity, _UNIT_1M)
                        yDPI = _convertToDPI(yDensity, _UNIT_1M)
                    else:  # no unit
                        xDPI = xDensity
                        yDPI = yDensity
                    break
                elif chunkType == b'IDAT':
                    break
                else:
                    try:
                        dataSize, = struct.unpack(">L", chunk[0:4])
                    except struct.error:
                        raise ValueError("Invalid PNG file")
                    chunkOffset += dataSize + 12
                    fhandle.seek(chunkOffset)
                    chunk = fhandle.read(17)
        # handle JPEGs
        elif size >= 2 and head.startswith(b'\377\330'):
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    if ftype == 0xe0:  # APP0 marker
                        fhandle.seek(7, 1)
                        unit, xDensity, yDensity = struct.unpack(">BHH", fhandle.read(5))
                        if unit == 1 or unit == 0:
                            xDPI = xDensity
                            yDPI = yDensity
                        elif unit == 2:
                            xDPI = _convertToDPI(xDensity, _UNIT_CM)
                            yDPI = _convertToDPI(yDensity, _UNIT_CM)
                        break
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
            except struct.error:
                raise ValueError("Invalid JPEG file")
        # handle JPEG2000s
        elif size >= 12 and head.startswith(b'\x00\x00\x00\x0cjP  \r\n\x87\n'):
            fhandle.seek(32)
            # skip JP2 image header box
            headerSize = struct.unpack('>L', fhandle.read(4))[0] - 8
            fhandle.seek(4, 1)
            foundResBox = False
            try:
                while headerSize > 0:
                    print("headerSize", headerSize)
                    boxHeader = fhandle.read(8)
                    boxType = boxHeader[4:]
                    print(boxType)
                    if boxType == 'res ':  # find resolution super box
                        foundResBox = True
                        headerSize -= 8
                        print("found res super box")
                        break
                    print("@1", boxHeader)
                    boxSize, = struct.unpack('>L', boxHeader[:4])
                    print("boxSize", boxSize)
                    fhandle.seek(boxSize - 8, 1)
                    headerSize -= boxSize
                if foundResBox:
                    while headerSize > 0:
                        boxHeader = fhandle.read(8)
                        boxType = boxHeader[4:]
                        print(boxType)
                        if boxType == 'resd':  # Display resolution box
                            print("@2")
                            yDensity, xDensity, yUnit, xUnit = struct.unpack(">HHBB", fhandle.read(10))
                            xDPI = _convertToDPI(xDensity, xUnit)
                            yDPI = _convertToDPI(yDensity, yUnit)
                            break
                        boxSize, = struct.unpack('>L', boxHeader[:4])
                        print("boxSize", boxSize)
                        fhandle.seek(boxSize - 8, 1)
                        headerSize -= boxSize
            except struct.error as e:
                print(e)
                raise ValueError("Invalid JPEG2000 file")
    return xDPI, yDPI


def hex2rgb(hex_code):
    assert isinstance(hex_code, str)
    hex_code = hex_code.replace("#", "")

    assert len(hex_code) == 6

    red, green, blue = bytes.fromhex(hex_code)
    return red, green, blue


def color_rgb2bgr(rgb):
    return rgb[2], rgb[1], rgb[0]


def color_bgr2rgb(bgr):
    return bgr[2], bgr[1], bgr[0]


def poly2box(polygon):
    x, y, w, h = cv2.boundingRect(polygon)
    return x, y, x + w, y + h


def show_image(img, windows_name, windows_size=SD_RESOLUTION, windows_mode=cv2.WINDOW_NORMAL, wait_time=1,
               key_press_exit="q"):
    """
    Show image in RGB format

    Parameters
    ----------
    img: numpy.ndarray
        image array

    windows_name: str
        Title of window

    windows_size: tuple[int, int]
        (Default: SD_RESOLUTION) size of window

    windows_mode: int
        (Default: cv2.WINDOW_NORMAL) Mode of window

    wait_time: int
        Block time. (-1: infinite)

    key_press_exit: str
        Key stop event.

    Returns
    -------
    bool
        True - Stop event from user
    """
    cv2.namedWindow(windows_name, windows_mode)
    cv2.imshow(windows_name, img[:, :, ::-1])
    cv2.resizeWindow(windows_name, *windows_size)

    if cv2.waitKey(wait_time) & 0xFF == ord(key_press_exit):
        cv2.destroyWindow(windows_name)
        return False
    return True


def destroy_windows(*windows_name):
    """
    Destroy windows if set. Else destroy all

    Parameters
    ----------
    windows_name: str
        List windows name. Empty same mean all windows.
    """
    if windows_name:
        for window_name in windows_name:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
    else:
        cv2.destroyAllWindows()


__all__ = ['hex2rgb', 'color_rgb2bgr', 'color_bgr2rgb', 'poly2box', 'show_image', 'image_info', 'image_dpi',
           'destroy_windows',
           'TIFF_FORMAT', 'JPEG2000_FORMAT', 'JPEG_FORMAT', 'PNG_FORMAT', 'GIF_FORMAT']
