from keras import Sequential
from keras.layers import Dense
from keras.optimizer_v1 import SGD


class FullyConnectedNet:

    @staticmethod
    def build(num_of_labels):
        # define the 3072-1024-512-|labels| architecture using Keras
        model = Sequential()
        model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
        model.add(Dense(512, activation="sigmoid"))
        model.add(Dense(num_of_labels, activation="softmax"))

        # compile the model using SGD as our optimizer and categorical
        # cross-entropy loss (you'll want to use binary_crossentropy
        # for 2-class classification)
        return model
