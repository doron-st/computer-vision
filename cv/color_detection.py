# import the necessary packages
import argparse
import cv2.cv2 as cv
import numpy as np

# construct the argument parse and parse the arguments
from cv import images
from cv.utils.numpy_utils import safe_add, safe_subtract

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file", default=f'{images}/lua_bao_bei.jpg')
    args = ap.parse_args()

    # load the image from disk
    image = cv.imread(args.image)
    image = cv.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # define the list of boundaries
    colors = [[231 * 179 / 239, 185, 174],  # pink
              [25 * 179 / 239, 223, 160],  # yellow
              [137 * 179 / 239, 137, 185],  # light-blue
              [77 * 179 / 239, 84, 78]  # light-blue
              ]
    colors = [np.array(c, dtype="uint8") for c in colors]
    print(colors)
    d = np.array([22, 100, 150], dtype="uint8")
    max_values = np.array([179, 255, 255], dtype="uint8")
    boundaries = [(safe_subtract(color, d), safe_add(color, d, max_values)) for color in colors]
    print(boundaries)
    # loop over the boundaries
    for (lower, upper) in boundaries:
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv.inRange(hsv, lower, upper)
        output = cv.bitwise_and(image, image, mask=mask)
        # show the images
        cv.imshow("images", np.hstack([image, output]))
        cv.waitKey(0)
