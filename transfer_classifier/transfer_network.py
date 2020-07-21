from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Activation, Flatten, \
    Dense, Dropout, ZeroPadding2D
from tensorflow.python.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np


def DenseNet(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Activation('sigmoid'))
    model.add(Dense(19))

    if weights_path:
        model.load_weights(weights_path)

    return model


def train_model(images, labels):
    print('[INFO] Train network')
    model = DenseNet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(images, labels, epochs=50, validation_split=0.25)
    model.save('model_vgg16CNN_bin_covid')
    model.summary()

    # Plot training & validation accuracy values
    plt.plot(np.arange(0, 50), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 50), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 50), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 50), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on COVID-19 Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()
