# CNN - Cancer Neural Network

import numpy as np

from keras.models import Sequential

from keras.layers import Dropout, Dense


def neural_network():

    data, labels = load_data("train_data.txt") #649
    validation_data, validation_labels = load_data("validation_data.txt") #50

    model = Sequential()

    model.add(Dense(64, input_dim=len(data[0]), activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(data, labels, nb_epoch=50, batch_size=10)

    print('\n')
    loss, accuracy = model.evaluate(validation_data, validation_labels, batch_size=50, verbose=1)
    print('loss: ' + str(round(loss, 4)))
    print('accuracy: ' + str(round(accuracy, 4)) + '\n')


def load_data(filename):
    file = open("../resources/" + str(filename), 'r')

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

    data = np.asarray(data)

    return data, labels


neural_network()
