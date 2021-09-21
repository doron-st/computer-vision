import argparse
import cv2.cv2 as cv2

from cvp.image.stitcher import Stitcher


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--first", required=True, help="path to the first image")
    ap.add_argument("-s", "--second", required=True, help="path to the second image")
    args = vars(ap.parse_args())
    # load the two images and resize them to have a width of 400 pixels
    # (for faster processing)
    image_a = cv2.imread(args["first"])
    image_b = cv2.imread(args["second"])
    # stitch the images together to create a panorama
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([image_a, image_b], show_matches=True)
    # show the images
    cv2.imshow("Image A", image_a)
    cv2.imshow("Image B", image_b)
    cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
