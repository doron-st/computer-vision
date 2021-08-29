import cv2.cv2 as cv

if __name__ == '__main__':

    # Load and show
    image = cv.imread("resources/images/apple.jpg")
    (h, w, d) = image.shape
    print("width={}, height={}, depth={}".format(w, h, d))

    # cv.imshow("Image", image)
    # cv.waitKey(0)

    # get pixel
    (B, G, R) = image[100, 50]
    print("R={}, G={}, B={}".format(R, G, B))

    # ROI
    roi = image[0:55, 105:220]
    # cv.imshow("ROI", roi)
    # cv.waitKey(0)

    # resizing
    resized = cv.resize(image, (600, 600))
    # cv.imshow("Fixed Resizing", resized)
    # cv.waitKey(0)

    # blurring
    # useful when reducing high frequency noise
    blurred = cv.GaussianBlur(resized, (51, 51), 0)
    # cv.imshow("Blurred", blurred)
    # cv.waitKey(0)

    # draw a 2px thick red rectangle surrounding the face
    output = image.copy()
    cv.rectangle(output, (105, 0), (220, 55), (255, 0, 170), 2)
    # cv.imshow("Rectangle", output)
    # cv.waitKey(0)

    # draw a blue 20px (filled in) circle on the image centered at
    cv.circle(output, (100, 150), 20, (231, 253, 255), -1)
    cv.circle(output, (130, 100), 10, (0, 0, 0), -1)
    # cv.imshow("Circle", output)
    # cv.waitKey(0)

    # draw a 5px thick red line from x=60,y=20 to x=400,y=200
    output = image.copy()
    cv.line(output, (20, 50), (250, 250), (0, 0, 255), 3)
    # cv.imshow("Line", output)
    # cv.waitKey(0)

    # draw green text on the image
    output = image.copy()
    cv.putText(output, "Apple!", (50, 100),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.imshow("Text", output)
    cv.waitKey(0)
