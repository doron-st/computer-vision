from cv.utils.image_utils import show, get_contours, scale
from cv.utils.parsing_utils import get_single_image_from_command_line
import cv2.cv2 as cv


def main():
    image = get_single_image_from_command_line()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 50, 255, cv.THRESH_BINARY)[1]
    eroded = cv.erode(thresh, None, iterations=2)
    diluted = cv.dilate(eroded, None, iterations=2)
    show(diluted)

    contours = get_contours(diluted)
    max_contour = max(contours, key=cv.contourArea)
    cv.drawContours(image, [max_contour], -1, (255, 255, 255), 2)

    ext_left = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
    ext_right = tuple(max_contour[max_contour[:, :, 0].argmax()][0])
    ext_top = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
    ext_bot = tuple(max_contour[max_contour[:, :, 1].argmax()][0])

    cv.circle(image, ext_left, 8, (255, 0, 0), -1)
    cv.circle(image, ext_right, 8, (0, 255, 0), -1)
    cv.circle(image, ext_top, 8, (0, 0, 255), -1)
    cv.circle(image, ext_bot, 8, (255, 255, 0), -1)
    show(image)


if __name__ == '__main__':
    main()
