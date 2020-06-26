import cv2
import numpy

BRIGHT_THRESHOLD = 0.5
DARK_THRESHOLD = 0.4


def is_bad_light_condition(img, dark_threshold=DARK_THRESHOLD, bright_threshold=BRIGHT_THRESHOLD):
    """
    Count bright pixel and dark pixel.
    Return:
        -1  :If dark_pixel / total_pixel > bright_threshold, it will be underexposed
         1  :If bright_pixel / total_pixel > dark_threshold, it will be overexposed
         0  :Good light condition
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_part = cv2.inRange(gray, 0, 30)
    bright_part = cv2.inRange(gray, 220, 255)

    # count pixel
    total_pixel = numpy.size(gray)
    dark_pixel = numpy.sum(dark_part > 0)
    bright_pixel = numpy.sum(bright_part > 0)

    if dark_pixel / total_pixel > bright_threshold:
        return -1
    if bright_pixel / total_pixel > dark_threshold:
        return 1
    return 0


def adjust_gamma(img, gamma):
    if gamma == 1:
        return img

    inv_gamma = 1.0 / gamma
    table = numpy.array([((i / 255.0) ** inv_gamma) * 255 for i in numpy.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


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


def highlight_center(img, radius, radius2=None, angle=0, thresh_hold=0.9):
    h, w = img.shape[:2]
    mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = mask * thresh_hold
    if radius2 is None or radius2 == radius:
        mask = cv2.circle(mask, (w//2, h//2), radius, color=255, thickness=-1)
    else:
        mask = cv2.ellipse(mask, (w//2, h//2), (radius, radius2), angle, 0, 360, 255, -1)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)/255
    mask = numpy.stack([mask, mask, mask], axis=2)
    img = numpy.array(img, dtype=numpy.float)
    img = img * mask
    return img.astype(numpy.uint8)
