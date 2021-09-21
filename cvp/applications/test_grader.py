import cv2.cv2 as cv2
import numpy as np
from imutils import contours

from cvp.applications.document_scanner import scan_document
from cvp.shape.countours import get_contours
from cvp.utils.image_utils import show
from cvp.utils.parsing_utils import get_single_image_from_command_line

# define the answer key which maps the question number to the correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}


def main():
    image = get_single_image_from_command_line()
    scanned_document = scan_document(image, color=True)
    all_contours = get_contours(scanned_document, 50, thresh_type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    grey = cv2.cvtColor(scanned_document, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey, 50, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    question_contours = []
    # loop over the contours
    for choice in all_contours:
        # compute the bounding box of the contour, then use the bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(choice)
        ar = w / float(h)
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and have an aspect ratio approximately equal to 1
        if w >= 30 and h >= 30 and 0.9 <= ar <= 1.1:
            question_contours.append(choice)

    question_contours = contours.sort_contours(question_contours, method="top-to-bottom")[0]
    correct_answers = 0
    # each question has 5 possible answers, to loop over the
    # question in batches of 5
    for (q, i) in enumerate(np.arange(0, len(question_contours), 5)):
        # sort the contours for the current question from
        # left to right, then initialize the index of the
        # bubbled answer
        choices = contours.sort_contours(question_contours[i:i + 5])[0]
        bubbled_index = -1
        bubbled_pixels = 0

        # loop over the sorted contours
        for (choice_index, choice) in enumerate(choices):
            # construct a mask that reveals only the current
            # "bubble" for the question
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [choice], -1, 255, -1)
            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            num_of_pixels = cv2.countNonZero(mask)
            # if the current total has a larger number of total
            # non-zero pixels, then we are examining the currently
            # bubbled-in answer
            if bubbled_index == -1 or num_of_pixels > bubbled_pixels:
                bubbled_index = choice_index
                bubbled_pixels = num_of_pixels

        # initialize the contour color and the index of the *correct* answer
        color = (0, 0, 255)  # red == mistake
        k = ANSWER_KEY[q]
        # check to see if the bubbled answer is correct
        if k == bubbled_index:
            color = (0, 255, 0)
            correct_answers += 1
        # draw the outline of the correct answer on the test
        cv2.drawContours(scanned_document, [choices[k]], -1, color, 3)
    # grab the test taker
    score = (correct_answers / 5.0) * 100
    print("[INFO] score: {:.2f}%".format(score))
    cv2.putText(scanned_document, "{:.2f}%".format(score), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    show(scanned_document, "Exam")


if __name__ == '__main__':
    main()
