import cv2.cv2 as cv
import imutils


def scale(image, ratio):
    return cv.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))


def show(image, name=''):
    cv.imshow(name, image)
    cv.waitKey(0)


def get_contours(threshold_image):
    contours = cv.findContours(threshold_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(contours)

