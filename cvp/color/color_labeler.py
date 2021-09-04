# import the necessary packages
from typing import Tuple, Optional

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2.cv2 as cv2

from cvp.utils.image_utils import show


class ColorLabeler:
    def __init__(self):
        # initialize the colors dictionary, containing the color
        # name as the key and the RGB tuple as the value
        colors = OrderedDict({
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (200, 200, 0),
            "orange": (250, 150, 50),
            "bluish-green": (0, 250, 100),
        })

        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []
        # loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(colors.items()):
            # update the L*a*b* array and the color names list
            self.lab[i] = rgb
            self.colorNames.append(name)
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def label(self, image, contour):
        # construct a mask for the contour, then compute the average L*a*b* value for the masked region
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        show(mask)
        mean = cv2.mean(image, mask=mask)[:3]
        # initialize the minimum distance found thus far
        min_dist: Tuple[float, Optional[int]] = (np.inf, None)
        # loop over the known L*a*b* color values
        for (i, row) in enumerate(self.lab):
            # compute the distance between the current L*a*b*
            # color value and the mean of the image
            d = dist.euclidean(row[0], mean)
            # if the distance is smaller than the current distance,
            # then update the bookkeeping variable
            if d < min_dist[0]:
                min_dist = (d, i)
        # return the name of the color with the smallest distance
        return self.colorNames[min_dist[1]]
