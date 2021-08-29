# import the necessary packages
from typing import List
import sys
import argparse
import imutils
import cv2.cv2 as cv2

from cv import images


def main(argv: List[str]):
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to input image", default=f'{images}/tetris.jpg')
    args = ap.parse_args(argv[1:])

    image = cv2.imread(args.image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    cv2.waitKey(0)

    # applying edge detection we can find the outlines of objects in images
    edged = cv2.Canny(gray, 30, 150)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)

    # threshold the image by setting all pixel values less than 225 to 255 (white; foreground)
    # and all pixel values >= 225 to 255(black; background), thereby segmenting the image
    thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)

    # we apply erosion to reduce the size of foreground objects
    mask = thresh.copy()
    mask = cv2.erode(mask, None, iterations=5)
    cv2.imshow("Eroded", mask)
    cv2.waitKey(0)

    # find contours (i.e., outlines) of the foreground objects in the thresholded image
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    output = image.copy()

    # loop over the contours
    for c in contours:
        # draw each contour on the output image with a 3px thick purple
        # outline, then display the output contours one at a time
        cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
        cv2.imshow("Contours", output)
        cv2.waitKey(0)

    # draw the total number of contours found in purple
    text = "I found {} objects!".format(len(contours))
    cv2.putText(output, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 159), 2)
    cv2.imshow("Contours", output)
    cv2.waitKey(0)


if __name__ == '__main__':
    main(sys.argv)
