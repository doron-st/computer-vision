import argparse
import cv2.cv2 as cv

from cvp.utils.image_utils import scale
import os


def get_single_image_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to the input image")
    return parser


def get_single_image_from_command_line():
    parser = get_single_image_parser()
    args = parser.parse_args()
    if not os.path.exists(args.image):
        raise RuntimeError(f'Could not find file {args.image}')
    image = cv.imread(args.image)
    while image.shape[0] > 1000:
        image = scale(image, 0.5)
    return image
