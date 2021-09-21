# import the necessary packages
import argparse
import cv2.cv2 as cv2
import os
import pickle
from tensorflow.keras.models import load_model


def parse_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_dir", required=True,
                    help="path to input image we are going to classify")
    ap.add_argument("-m", "--model", required=True,
                    help="path to trained Keras model")
    ap.add_argument("-l", "--label_bin", required=True,
                    help="path to label binarizer")
    ap.add_argument("-w", "--width", type=int, default=32,
                    help="target spatial dimension width")
    ap.add_argument("-e", "--height", type=int, default=32,
                    help="target spatial dimension height")
    ap.add_argument("-f", "--flatten", type=int, default=-1,
                    help="whether or not we should flatten the image")
    args = ap.parse_args()
    return args


def main():
    args = parse_args()

    # load the model and label binarizer
    print("[INFO] loading network and label binarizer...")
    model = load_model(args.model)
    lb = pickle.loads(open(args.label_bin, "rb").read())

    for image_file in os.listdir(args.image_dir):
        image = cv2.imread(os.path.join(args.image_dir, image_file))
        output = image.copy()
        image = process_image(image, args)
        # make a prediction on the image
        label, confidence = classify(image, lb, model)
        show_result(label, output, confidence)


def process_image(image, args):
    image = cv2.resize(image, (args.width, args.height))
    # scale the pixel values to [0, 1]
    image = image.astype("float") / 255.0
    # check to see if we should flatten the image and add a batch dimension
    if args.flatten > 0:
        image = image.flatten()
        image = image.reshape((1, image.shape[0]))
    # otherwise, we must be working with a CNN -- don't flatten the
    # image, simply add the batch dimension
    else:
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    return image


def classify(image, lb, model):
    preds = model.predict(image)
    # find the class label index with the largest corresponding
    # probability
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]
    confidence = preds[0][i] * 100
    return label, confidence


def show_result(label, output, confidence):
    # draw the class label + probability on the output image
    text = "{}: {:.2f}%".format(label, confidence)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the output image
    cv2.imshow("Image", output)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
