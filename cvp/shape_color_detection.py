from cvp.color.color_labeler import ColorLabeler
from cvp.shape.countours import get_contours, find_center
from cvp.shape.shape_detector import detect_shape
from cvp.utils.image_utils import show
from cvp.utils.parsing_utils import get_single_image_from_command_line
import cv2.cv2 as cv


def main():
    image = get_single_image_from_command_line()
    contours = get_contours(image, 80)
    lab_image = cv.cvtColor(cv.blur(image, (5, 5)), cv.COLOR_BGR2LAB)
    color_labeler = ColorLabeler()

    # loop over the contours

    for c in contours:
        # compute the center of the contour
        center = find_center(c)
        # detect the shape of the contour and label the color
        shape = detect_shape(c)
        color = color_labeler.label(lab_image, c)
        text = "{} {}".format(color, shape)
        cv.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv.putText(image, text, center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        show(image)


if __name__ == '__main__':
    main()
