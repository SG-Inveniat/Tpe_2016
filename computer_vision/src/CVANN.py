# CVANN - Computer Vision Artificial Neural Network

import os
import numpy as np

from keras.models import Sequential
from keras.layers import MaxPooling2D, Convolution2D
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def main():
    train = load_data('AV_train')
    validation = load_data('AV_validation')
    model = neural_network(train, validation)  # train, validation


def neural_network(train, validation, nb_epoch=5, batch_size=40, activations=('relu', 'relu', 'softmax'),
                   nb_filters=30, img_dimensions=(120, 90), pool_dimensions=(8, 6), conv_dimensions=(12, 9),
                   optimizer='rmsprop'):

    model = Sequential()
    model.add(Convolution2D(nb_filter=nb_filters, nb_row=conv_dimensions[0], nb_col=conv_dimensions[1],
                            input_shape=(1, img_dimensions[0], img_dimensions[1]), activation=activations[0]))

    model.add(MaxPooling2D(pool_size=pool_dimensions))
    model.add(Dropout(0.5))
    model.add(Flatten())  # converts our 2D feature maps to 1D feature vectors
    model.add(Dense(128, activation=activations[1]))
    model.add(Dropout(0.3))
    model.add(Dense(train[2], activation=activations[2]))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x=train[0], y=train[1], nb_epoch=nb_epoch, batch_size=batch_size)

    print('\n')
    loss, accuracy = model.evaluate(x=validation[0], y=validation[1], batch_size=batch_size, verbose=1)
    print('loss: ' + str(round(loss, 4)))
    print('accuracy: ' + str(round(accuracy, 4)) + '\n')

    return model


def load_data(img_dir_name, img_dimensions=(120, 90)):

    print('\nLoading data: "' + str(img_dir_name) + '"')

    directory = '..' + os.sep + 'resources' + os.sep + str(img_dir_name)
    sub_dir_names = [file for file in os.listdir(directory) if not file.startswith('.')]

    # calculate the total number of images | initialize the dictionary mapping the names to integers
    total_num_images = 0
    i = 0
    dictionary = {}

    for sub_dir in sub_dir_names:
        sub_dir_path = directory + os.sep + sub_dir
        image_files = [file for file in os.listdir(sub_dir_path) if not file.startswith('.')]
        total_num_images += len(image_files)

        dictionary[sub_dir] = i
        i += 1

    print('Loading ' + str(total_num_images) + ' images\n')

    data = []
    labels = np.zeros((total_num_images, 1)).astype('int')
    i = 0
    for sub_dir in sub_dir_names:
        print('current directory of images: ' + sub_dir)

        label = int(dictionary[sub_dir])  # the 'label' associated to this folder
        sub_dir_path = directory + os.sep + sub_dir
        img_files = [file for file in os.listdir(sub_dir_path) if not file.startswith('.')]

        for file in img_files:
            file_path = sub_dir_path + os.sep + file
            img = load_img(file_path, grayscale=True, target_size=img_dimensions)
            # img = sep_grayscale_intervals(img, num_intervals=8)  # testing phase
            img_array = img_to_array(img)

            labels = np.insert(labels, i, label, axis=0)  # insert the label at position i
            labels = np.delete(labels, -1, axis=0)  # remove the one too many element
            data.append(img_array)
            i += 1

    data = np.asarray(data)
    data /= 255
    nb_classes = len(dictionary)
    labels = to_categorical(labels, nb_classes=nb_classes)

    return data, labels, nb_classes


def sep_grayscale_intervals(img, num_intervals=4, output_path=None):

    img_array = img_to_array(img)

    for img in range(len(img_array)):
        for i in range(len(img_array[img])):
            for j in range(len(img_array[img][i])):

                pixel = img_array[img][i][j]
                new_value = pixel

                for interval in range(num_intervals):
                    interval_max = 255-((256/num_intervals)*interval)
                    if pixel < interval_max:
                        new_value = interval_max-(256/num_intervals)

                img_array[img][i][j] = new_value

    if output_path:
        img.save(output_path, 'JPEG')

    return array_to_img(img_array)


if __name__ == '__main__':
    main()
