# import the necessary packages
from cvp.shape.countours import get_contours
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2.cv2 as cv2
import numpy as np


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True, help="path to output directory")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    ap.add_argument("-f", "--fps", type=int, default=20, help="FPS of output video")
    ap.add_argument("-c", "--codec", type=str, default="MJPG", help="codec of output video")
    ap.add_argument("-b", "--buffer-size", type=int, default=32,
                    help="buffer size of video clip writer")
    args = vars(ap.parse_args())

    # initialize the video stream and allow the camera sensor to
    # warmup
    print("[INFO] warming up camera...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    print("[INFO] done")

    # define the lower and upper boundaries of the "green" ball in
    # the HSV color space
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)
    # initialize key clip writer and the consecutive number of
    # frames that have *not* contained any action
    # kcw = KeyClipWriter(buf_size=args["buffer_size"])
    consecFrames = 0
    firstFrame = None

    # loop over the frames of the video
    while True:
        # time.sleep(0.1)
        # grab the current frame and initialize the occupied/unoccupied
        # text
        frame = vs.read()
        frame = frame if args.get("video", None) is None else frame[1]
        text = "Unoccupied"
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if frame is None:
            break
        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        cnts = get_contours(frameDelta, threshold=25, thresh_type=cv2.THRESH_BINARY, is_color=False)
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"

        # draw the text and timestamp on the frame
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", np.hstack([frame, cv2.cvtColor(frameDelta, cv2.COLOR_GRAY2BGR)]))
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the loop
        if key == ord("q"):
            break
    # cleanup the camera and close any open windows
    vs.stop() if args.get("video", None) is None else vs.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()