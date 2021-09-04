from cvp.utils.image_utils import show
from cvp.shape.countours import get_contours_from_thresh, get_extreme_points
from cvp.utils.parsing_utils import get_single_image_from_command_line
import cv2.cv2 as cv


def main():
    image = get_single_image_from_command_line()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 50, 255, cv.THRESH_BINARY)[1]
    eroded = cv.erode(thresh, None, iterations=2)
    diluted = cv.dilate(eroded, None, iterations=2)
    show(diluted)

    contours = get_contours_from_thresh(diluted)
    max_contour = max(contours, key=cv.contourArea)
    cv.drawContours(image, [max_contour], -1, (255, 255, 255), 2)

    ext_bot, ext_left, ext_right, ext_top = get_extreme_points(max_contour)

    cv.circle(image, ext_left, 8, (255, 0, 0), -1)
    cv.circle(image, ext_right, 8, (0, 255, 0), -1)
    cv.circle(image, ext_top, 8, (0, 0, 255), -1)
    cv.circle(image, ext_bot, 8, (255, 255, 0), -1)
    show(image)


if __name__ == '__main__':
    main()
