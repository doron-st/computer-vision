# import the necessary packages
import cv2.cv2 as cv2
import imutils

# construct the argument parse and parse the arguments
from cv.utils.image_utils import show
from cv.utils.parsing_utils import get_single_image_parser


def main():
    parser = get_single_image_parser()
    args = parser.parse_args()
    # load the image, convert it to grayscale, blur it slightly,
    # and threshold it
    image = cv2.imread(args.image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    show(blurred)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    show(thresh)

    # find contours in the thresholded image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # loop over the contours
    for c in contours:
        # compute the center of the contour
        moments = cv2.moments(c)
        if moments['m00'] == 0:
            continue
        c_x = int(moments['m10'] / moments['m00'])
        c_y = int(moments['m01'] / moments['m00'])
        # draw the contour and center of the shape on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (c_x, c_y), 7, (255, 255, 255), -1)
        cv2.putText(image, "center", (c_x - 20, c_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # show the image
        cv2.imshow("Image", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
