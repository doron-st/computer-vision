from typing import Optional, Tuple

import imutils
from cv2 import cv2 as cv

from cvp.utils.image_utils import show


def get_contours(image, threshold: int, thresh_type=cv.THRESH_BINARY, is_color=True, do_show=False):
    if is_color:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh_image = cv.threshold(blurred, threshold, 255, thresh_type)[1]
    if do_show:
        show(thresh_image)
    contours = get_contours_from_thresh(thresh_image)
    return contours


def get_edged(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    edged = cv.Canny(blurred, 75, 200)
    return edged


def get_contours_from_thresh(threshold_image):
    contours = cv.findContours(threshold_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(contours)


def find_center(contour) -> Optional[Tuple[int, int]]:
    moments = cv.moments(contour)
    if moments['m00'] == 0:
        return None
    c_x = int(moments['m10'] / moments['m00'])
    c_y = int(moments['m01'] / moments['m00'])
    return c_x, c_y


def get_extreme_points(max_contour):
    ext_left = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
    ext_right = tuple(max_contour[max_contour[:, :, 0].argmax()][0])
    ext_top = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
    ext_bot = tuple(max_contour[max_contour[:, :, 1].argmax()][0])
    return ext_bot, ext_left, ext_right, ext_top