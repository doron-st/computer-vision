import cv2.cv2 as cv

from cvp.utils.image_utils import show


def apply_threshold(image,
                    threshold: int,
                    thresh_type=cv.THRESH_BINARY,
                    is_color=True,
                    blur_kernel_size=5,
                    dilution_iterations=0,
                    eroding_iterations=0,
                    do_show=False):
    if is_color:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    blurred = cv.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    thresh_image = cv.threshold(blurred, threshold, 255, thresh_type)[1]
    eroded = cv.erode(thresh_image, None, iterations=eroding_iterations)
    diluted = cv.dilate(eroded, None, iterations=dilution_iterations)
    if do_show:
        show(diluted)
    return diluted
