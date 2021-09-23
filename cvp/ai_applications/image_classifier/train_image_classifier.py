# set the matplotlib backend so figures can be saved in the background
import argparse
import cv2.cv2 as cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from imutils import paths
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from cvp.ai_applications.image_classifier.fully_connected_net import FullyConnectedNet
from cvp.ai_applications.image_classifier.small_vgg_net import SmallVGGNet
from cvp.utils.io_utils import write_pickle

matplotlib.use("Agg")


# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 50
MODEL_NAME = 'small_vgg'
# MODEL_NAME = 'fully_connected'


def parse_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset of images")
    ap.add_argument("-o", "--output_dir", required=True,
                    help="path to output dir")
    ap.add_argument("-m", "--model", required=True,
                    help="filename of trained model")
    ap.add_argument("-l", "--label_bin", required=True,
                    help="filename of label binarizer")
    ap.add_argument("-p", "--plot", required=True,
                    help="filename of output accuracy/loss plot")
    return ap.parse_args()


def load_data(dataset):
    # initialize the data and labels
    print("[INFO] loading images...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    image_paths = sorted(list(paths.list_images(dataset)))
    random.seed(42)
    random.shuffle(image_paths)
    # loop over the input images
    for image_path in image_paths:
        # load the image, resize the image to be 32x32 pixels (ignoring
        # aspect ratio), flatten the image into 32x32x3=3072 pixel image
        # into a list, and store the image in the data list
        image = cv2.imread(image_path)
        if MODEL_NAME == 'fully_connected':
            image = cv2.resize(image, (32, 32)).flatten()
        else:
            image = cv2.resize(image, (64, 64))
        data.append(image)
        # extract the class label from the image path and update the
        # labels list
        label = image_path.split(os.path.sep)[-2]
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    return data, labels


def plot_eval(H, output_file):
    # plot the training loss and accuracy
    n = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(n, H.history["loss"], label="train_loss")
    plt.plot(n, H.history["val_loss"], label="val_loss")
    plt.plot(n, H.history["accuracy"], label="train_acc")
    plt.plot(n, H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(output_file)


def main():
    args = parse_args()
    data, labels = load_data(args.dataset)
    print(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

    lb = LabelBinarizer()
    train_y = lb.fit_transform(train_y)
    test_y = lb.transform(test_y)
    num_of_categories = len(lb.classes_)

    if MODEL_NAME == 'fully_connected':
        model = FullyConnectedNet.build(num_of_categories)
    elif MODEL_NAME == 'small_vgg':
        model = SmallVGGNet.build(64, 64, 3, num_of_categories)
    else:
        raise RuntimeError('undefined MODEL_NAME (fully_connected | small_vgg)')

    opt = SGD(learning_rate=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("[INFO] training network...")
    h = model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), epochs=EPOCHS, batch_size=32)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(x=test_x, batch_size=32)
    print(classification_report(test_y.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=lb.classes_))
    os.makedirs(args.output_dir, exist_ok=True)
    plot_eval(h, os.path.join(args.output_dir, args.plot))
    model.save(os.path.join(args.output_dir, args.model), save_format="h5")
    write_pickle(lb, os.path.join(args.output_dir, args.label_bin))


if __name__ == '__main__':
    main()
