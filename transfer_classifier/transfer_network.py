from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Activation, Conv2D, MaxPool2D
from tensorflow.python.keras.models import Model

from utils import plot_binary_metric


def DenseNet(weights_path=None):
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


def train_model(images, labels, epochs=10):
    print('[INFO] Train network')
    model = DenseNet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(images, labels, epochs=epochs, validation_split=0.25, workers=12)
    print('[INFO] Save network')
    model.save('model_multipleCNN_bin_covid')
    model.summary()
    plot_binary_metric(epochs, history, 'multiple_model.pdf')
    return model


def train_binary_using_pretrained_model(images, labels, model, epochs=10):
    inputs = Input(shape=(224, 224, 1))
    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[9:]:
        layer.trainable = True
    x = model(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(images, labels, epochs=epochs, validation_split=0.1, workers=12)
    model.summary()
    print("[INFO] evaluating after fine-tuning network...")
    plot_binary_metric(epochs, history, 'binary_pretrained_model.pdf')


def train_using_pretrained_model(images, labels, model, epochs=10):
    print("[INFO] evaluating after fine-tuning network head...")

    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[9:]:
        layer.trainable = True

    for layer in model.layers:
        print("{}: {}".format(layer, layer.trainable))

    print("[INFO] re-compiling model...")
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["binary_accuracy"])

    history = model.fit(images, labels, epochs=epochs, validation_split=0.1)

    print("[INFO] evaluating after fine-tuning network...")
    plot_binary_metric(epochs, history, 'multiple_pretrained_model.pdf')

    print("[INFO] Save network...")
    model.save('model_finetuneCNN_bin_covid')
