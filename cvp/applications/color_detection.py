# import the necessary packages
import cv2.cv2 as cv
import numpy as np

# construct the argument parse and parse the arguments
from cvp.utils.numpy_utils import safe_add, safe_subtract
from cvp.utils.parsing_utils import get_single_image_from_command_line

if __name__ == '__main__':
    image = get_single_image_from_command_line()
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # define the list of boundaries
    colors = [[231, 185, 174],  # pink
              [25, 223, 160],  # yellow
              [137, 137, 185],  # light-blue
              [77, 84, 78],  # light-blue
              [5, 132, 204]  # body
              ]

    # transform from paint to openCV hue scale
    r = 179 / 239
    colors = np.array([[int(a * b) for a, b in zip(color, [r, 1, 1])] for color in colors], dtype="uint8")

    d = np.array([22, 100, 150], dtype="uint8")
    max_values = np.array([179, 255, 255], dtype="uint8")
    boundaries = [(safe_subtract(color, d), safe_add(color, d, max_values)) for color in colors]

    # loop over the boundaries
    for (lower, upper) in boundaries:
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv.inRange(hsv, lower, upper)
        output = cv.bitwise_and(image, image, mask=mask)
        # show the images
        cv.imshow("images", np.hstack([image, output]))
        cv.waitKey(0)
