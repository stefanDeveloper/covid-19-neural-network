from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Activation, Conv2D, MaxPool2D

from utils import plot_binary_metric


def CNN(weights_path=None):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 1)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15))
    model.add(Activation('sigmoid'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def train_model(images_first_train, labels_first_train, images_second_train, labels_second_train, epochs=10):
    print('[INFO] Train network')
    model = CNN()

    opt = SGD(lr=1e-4, momentum=0.9)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"])

    print("[INFO] training head...")

    history = model.fit(images_first_train, labels_first_train, epochs=epochs, validation_split=0.25)

    plot_binary_metric(epochs, history)

    print("[INFO] evaluating after fine-tuning network head...")

    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[8:]:
        layer.trainable = True

    for layer in model.layers:
        print("{}: {}".format(layer, layer.trainable))

    print("[INFO] re-compiling model...")
    opt = SGD(lr=1e-4, momentum=0.9)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"])

    history = model.fit(images_second_train, labels_second_train, epochs=30, validation_split=0.25)

    print("[INFO] evaluating after fine-tuning network...")
    plot_binary_metric(30, history)

    print("[INFO] Save network...")
    model.save('model_vgg16CNN_bin_covid')
