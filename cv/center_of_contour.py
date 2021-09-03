# import the necessary packages
import cv2.cv2 as cv
import imutils

# construct the argument parse and parse the arguments
from cv.utils.image_utils import show, get_contours
from cv.utils.parsing_utils import get_single_image_parser, get_single_image_from_command_line


def main():
    image = get_single_image_from_command_line()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    show(blurred)
    thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]
    show(thresh)

    # find contours in the thresholded image
    contours = get_contours(thresh)

    # loop over the contours
    for c in contours:
        # compute the center of the contour
        moments = cv.moments(c)
        if moments['m00'] == 0:
            continue
        c_x = int(moments['m10'] / moments['m00'])
        c_y = int(moments['m01'] / moments['m00'])
        # draw the contour and center of the shape on the image
        cv.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv.circle(image, (c_x, c_y), 7, (255, 255, 255), -1)
        cv.putText(image, "center", (c_x - 20, c_y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # show the image
        cv.imshow("Image", image)
        cv.waitKey(0)


if __name__ == '__main__':
    main()
