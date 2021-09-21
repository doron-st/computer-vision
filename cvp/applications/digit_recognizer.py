from imutils import contours
from imutils.perspective import four_point_transform

from cvp.image.threshold import apply_threshold
from cvp.shape.countours import get_contours, get_contours_from_thresh
from cvp.shape.shape_detector import detect_shape
from cvp.utils.image_utils import show, scale
from cvp.utils.parsing_utils import get_single_image_from_command_line

import cv2.cv2 as cv2
import numpy as np

debug = True


def main():
    # construct the argument parser and parse the arguments
    image = get_single_image_from_command_line()

    if debug:
        show(image)
    contours_1 = get_contours(image, 100, thresh_type=cv2.THRESH_BINARY_INV)

    lcd_contour = contours_1[0]
    shape_name, screen_shape = detect_shape(lcd_contour)

    if shape_name != 'rectangle':
        raise RuntimeError('Could not detect lcs screen')

    cv2.drawContours(image, [lcd_contour], -1, (240, 0, 159), 3)
    # show(image, 'lcd contour')

    lcd_screen = four_point_transform(image, screen_shape.reshape(4, 2))
    if debug:
        show(lcd_screen, 'lcd screen')

    lcd_screen_thresh = apply_threshold(lcd_screen,
                                        threshold=50,
                                        thresh_type=cv2.THRESH_BINARY_INV,
                                        eroding_iterations=2,
                                        dilution_iterations=3)

    contours_2 = get_contours_from_thresh(lcd_screen_thresh)

    digit_contours = []
    for c in contours_2:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 15 and h > 15:
            digit_contours.append(c)

    digit_contours = contours.sort_contours(digit_contours, method="left-to-right")[0]
    inferred_digits = []
    for digit in digit_contours:
        cv2.drawContours(lcd_screen, [digit], -1, (240, 0, 159), 1)
        if debug:
            show(lcd_screen, 'lcd contour')
        (x, y, w, h) = cv2.boundingRect(digit)
        digit_roi = lcd_screen_thresh[y:y + h, x:x + w]
        max_cor = 0
        matching_digit = -1
        best_corr_matrix = None
        for i in range(10):
            digit_template = get_digit(i, w, h).astype(np.int8) * 2 - 1
            digit_roi_minus_1_1 = (digit_roi - 127.5) / 127.5
            corr = np.multiply(digit_roi_minus_1_1, digit_template)
            corr_score = corr.sum()
            # print(f'corr to {i} is {corr_score}')
            if corr_score > max_cor:
                max_cor = corr_score
                matching_digit = i
                best_corr_matrix = corr
        inferred_digits.append(matching_digit)
        if debug:
            show(scale((best_corr_matrix + 1) * 255, 3), 'best_corr')
        cv2.putText(lcd_screen, str(matching_digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 159), 2)
        if debug:
            show(lcd_screen)
    print(f'Temperature is: {inferred_digits[0]}{inferred_digits[1]}.{inferred_digits[2]} Celsius')


def get_digit(digit, w, h):
    digit_image = None
    if digit == 1:
        digit_image = np.array([[1, 0, 0, 0],
                                [1, 0, 0, 0],
                                [1, 0, 0, 0],
                                [1, 0, 0, 0],
                                [1, 0, 0, 0]
                                ], dtype='uint8')
    elif digit == 2:
        digit_image = np.array([[1, 1, 1, 1],
                                [0, 0, 0, 1],
                                [1, 1, 1, 1],
                                [1, 0, 0, 0],
                                [1, 1, 1, 1]
                                ], dtype='uint8')
    elif digit == 3:
        digit_image = np.array([[1, 1, 1, 1],
                                [0, 0, 0, 1],
                                [1, 1, 1, 1],
                                [0, 0, 0, 1],
                                [1, 1, 1, 1]
                                ], dtype='uint8')
    elif digit == 4:
        digit_image = np.array([[1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [1, 1, 1, 1],
                                [0, 0, 0, 1],
                                [0, 0, 0, 1]
                                ], dtype='uint8')
    elif digit == 5:
        digit_image = np.array([[1, 1, 1, 1],
                                [1, 0, 0, 0],
                                [1, 1, 1, 1],
                                [0, 0, 0, 1],
                                [1, 1, 1, 1]
                                ], dtype='uint8')
    elif digit == 6:
        digit_image = np.array([[1, 1, 1, 1],
                                [1, 0, 0, 0],
                                [1, 1, 1, 1],
                                [1, 0, 0, 1],
                                [1, 1, 1, 1]
                                ], dtype='uint8')
    elif digit == 7:
        digit_image = np.array([[1, 1, 1, 1],
                                [0, 0, 0, 1],
                                [0, 0, 0, 1],
                                [0, 0, 0, 1],
                                [0, 0, 0, 1]
                                ], dtype='uint8')
    elif digit == 8:
        digit_image = np.array([[1, 1, 1, 1],
                                [1, 0, 0, 1],
                                [1, 1, 1, 1],
                                [1, 0, 0, 1],
                                [1, 1, 1, 1]
                                ], dtype='uint8')
    elif digit == 9:
        digit_image = np.array([[1, 1, 1, 1],
                                [1, 0, 0, 1],
                                [1, 1, 1, 1],
                                [0, 0, 0, 1],
                                [1, 1, 1, 1]
                                ], dtype='uint8')
    elif digit == 0:
        digit_image = np.array([[1, 1, 1, 1],
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [1, 0, 0, 1],
                                [1, 1, 1, 1]
                                ], dtype='uint8')

    return cv2.resize(digit_image, (w, h))


if __name__ == '__main__':
    main()
