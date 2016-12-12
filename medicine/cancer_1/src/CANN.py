# CANN - Cancer Artificial Neural Network

import os

import numpy as np

from keras.models import Sequential
from keras.layers import Dropout, Dense


def neural_network():
    # define the general structure of the neural network, train, and assert its real-life performance based on the
    # accuracy on the validation data

    # load the data onto the system
    data, labels = load_data("train_data.txt")  # 649
    validation_data, validation_labels = load_data("validation_data.txt")  # 50

    model = Sequential()    # define model variable as Keras Sequential() object

    model.add(Dense(64, input_dim=len(data[0]), activation='sigmoid'))  # add the different layers to the model
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dropout(0.25))    # dropout
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',   # compile the given binary classificaiton model with rmsprop as
                  optimizer='rmsprop',          # optimizer, more about modern optimizers on:
                  metrics=['accuracy'])         # http://sebastianruder.com/optimizing-gradient-descent/

    model.fit(data, labels, nb_epoch=50, batch_size=10)  # train the model

    # evaluate real-life performance via validation data that the network has never seen
    print('\n')
    loss, accuracy = model.evaluate(validation_data, validation_labels, batch_size=50, verbose=1)
    print('loss: ' + str(round(loss, 4)))   # print loss
    print('accuracy: ' + str(round(accuracy, 4)) + '\n')    # print % accuracy


def load_data(filename):
    # load data from 'filename' located in the ../resources folder and convert it to keras-compatible format

    file = open(".." + os.sep + "resources" + os.sep + str(filename), 'r')

    data = []
    labels = []

    for line in file:

        low = 0
        high = 0
        arr = []

        for char in line:
            high += 1
            if char == ',':
                arr.append(int(line[low:high-1]))
                low = high

        data.append(np.asarray(arr[1:-1]))

        if arr[-1] == 2:
            labels.append(0)
        else:
            labels.append(1)

    data = np.asarray(data)  # variable 'data' is a two-dimensional numpy array

    return data, labels  # return the data and the corresponding labels (Malignant/Benign)


neural_network()    # run the network
