from tensorflow import keras
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Conv3D, concatenate, UpSampling2D, Activation, Flatten, \
    Dense, Dropout
from tensorflow.python.keras.models import Model, Sequential
import matplotlib.pyplot as plt
import numpy as np


def DenseNet():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def train_model(images, labels):
    print('[INFO] Train network')
    model = DenseNet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(images, labels, epochs=50, verbose=1, validation_split=0.25, )
    model.save('model_simpleCNN_bin_covid')

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
