# import the necessary packages
import cv2.cv2 as cv2
import imutils

from cvp.utils.image_utils import show
from cvp.utils.parsing_utils import get_single_image_from_command_line


def main():
    # construct the argument parser and parse the arguments
    image = get_single_image_from_command_line()
    show(image, 'Image')

    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show(gray, 'Gray-scale')

    # applying edge detection we can find the outlines of objects in images
    edged = cv2.Canny(gray, 30, 150)
    show(edged, 'Edged')

    # threshold the image by setting all pixel values less than 225 to 255 (white; foreground)
    # and all pixel values >= 225 to 255(black; background), thereby segmenting the image
    thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]
    show(thresh, 'Thresh')

    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    show(adaptive_thresh, "Adaptive thresh")

    # we apply erosion to reduce the size of foreground objects
    eroded = thresh.copy()
    eroded = cv2.erode(eroded, None, iterations=2)
    show(eroded, 'Eroded')

    # we apply dilution to reduce the size of background objects
    dilated = eroded.copy()
    dilated = cv2.dilate(dilated, None, iterations=2)
    show(dilated, 'Dilated')

    # find contours (i.e., outlines) of the foreground objects in the thresholded image
    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    output = image.copy()

    # loop over the contours
    for c in contours:
        # draw each contour on the output image with a 3px thick purple
        # outline, then display the output contours one at a time
        cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
        show(output, 'Contours')

    # draw the total number of contours found in purple
    text = "I found {} objects!".format(len(contours))
    cv2.putText(output, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 159), 2)
    show(output, 'Contours')


if __name__ == '__main__':
    main()
