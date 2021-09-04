import cv2.cv2 as cv

from cvp.utils.image_utils import show
from cvp.shape.countours import get_contours, find_center
from cvp.utils.parsing_utils import get_single_image_from_command_line


def main():
    image = get_single_image_from_command_line()
    contours = get_contours(image, 80)
    for c in contours:
        cv.drawContours(image, [c], -1, (255, 255, 255), 2)
        center = find_center(c)
        if center is None:
            continue
        shape = detect_shape(c)
        cv.putText(image, shape, center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print(shape)
        show(image)


def detect_shape(contour) -> str:
    perimeter = cv.arcLength(contour, True)
    approx_contour = cv.approxPolyDP(contour, 0.04 * perimeter, True)

    # if the shape is a triangle, it will have 3 vertices
    if len(approx_contour) == 3:
        shape = "triangle"
    # if the shape has 4 vertices, it is either a square or a rectangle
    elif len(approx_contour) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv.boundingRect(approx_contour)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx_contour) == 5:
        shape = "pentagon"
    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"
    # return the name of the shape
    return shape


if __name__ == '__main__':
    main()
