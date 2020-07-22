from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Activation, Flatten, \
    Dense, Dropout
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import Adam

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
    plot_binary_metric(epochs, history)
    return model


def train_using_pretrained_model(images, labels, base_model, epochs=10):
    print("[INFO] evaluating after fine-tuning network head...")

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[9:]:
        layer.trainable = True

    for layer in base_model.layers:
        print("{}: {}".format(layer, layer.trainable))

    print("[INFO] re-compiling model...")
    opt = Adam(lr=1e-5, momentum=0.9)
    base_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"])

    history = base_model.fit(images, labels, epochs=epochs, validation_split=0.1)

    print("[INFO] evaluating after fine-tuning network...")
    plot_binary_metric(epochs, history)

    print("[INFO] Save network...")
    base_model.save('model_finetuneCNN_bin_covid')
