from typing import Optional

from imutils.perspective import four_point_transform

from cvp.shape.countours import get_edged, get_contours, get_contours_from_thresh
from cvp.shape.shape_detector import detect_shape
from cvp.utils.image_utils import show
from cvp.utils.parsing_utils import get_single_image_from_command_line
import cv2.cv2 as cv
from skimage.filters import threshold_local
import numpy as np


def main():
    image = get_single_image_from_command_line()
    show(image)
    edged = get_edged(image)
    show(edged, 'Edges')
    contours = get_contours_from_thresh(edged)
    largest_contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

    screen_contour: Optional[np.array] = None
    for c in largest_contours:
        name, shape = detect_shape(c)
        if name == 'rectangle' or name == 'square':
            screen_contour = shape
            break

    cv.drawContours(image, [screen_contour], -1, (0, 255, 0), 2)
    show(image, "Outline")

    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(image, screen_contour.reshape(4, 2))
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    thresh = threshold_local(warped, 11, offset=10, method="gaussian")
    warped: np.ndarray = (warped > thresh)
    warped = warped.astype("uint8") * 255
    # show the original and scanned images
    show(warped, 'Scanned')


if __name__ == '__main__':
    main()
