import cv2


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


def show_image(img, windows_name, windows_size=(800, 600), windows_mode=cv2.WINDOW_NORMAL, show_time=1,
               key_press_exit="q"):
    """
    Show image
    - Options:
        show_time: in milliseconds
    :return: boolean, True - Stop event from user
    """
    cv2.namedWindow(windows_name, windows_mode)
    cv2.imshow(windows_name, img)
    cv2.resizeWindow(windows_name, *windows_size)

    if cv2.waitKey(show_time) & 0xFF == ord(key_press_exit):
        cv2.destroyWindow(windows_name)
        return False
    return True


__all__ = ['hex2rgb', 'color_rgb2bgr', 'color_bgr2rgb', 'poly2box', 'show_image']
