import cv2.cv2 as cv


def scale(image, ratio):
    return cv.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))


def show(image, name=''):
    cv.imshow(name, image)
    cv.waitKey(0)


