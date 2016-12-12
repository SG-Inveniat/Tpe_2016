# CNN - Cancer Neural Network

import numpy as np

from keras.models import Sequential

from keras.layers import Dropout, Dense


def neural_network():

    data, labels = load_data('train_data.txt') #519
    validation_data, validation_labels = load_data('validation_data.txt') #50

    model = Sequential()

    model.add(Dense(64, input_dim=len(data[0]), activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))  # relu, softmax aren't functional at the moment
    model.add(Dropout(0.25))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(data, labels, nb_epoch=100, batch_size=10)

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
                arr.append(line[low: high-1])
                low = high

        arr_data = arr[2:-1]
        temp = []
        for item in arr_data:
            temp.append(round(float(item), 2))

        data.append(np.asarray(temp))

        if str(arr[1]) == 'M':
            labels.append(1)
        else:
            labels.append(0)

    data = np.asarray(data)

    return data, labels


neural_network()
