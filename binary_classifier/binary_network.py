from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, \
    Dense, Dropout
from tensorflow.python.keras.models import Sequential

from utils import plot_metric


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


def train_model(images, labels, epochs=10):
    print('[INFO] Train network')
    model = DenseNet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(images, labels, epochs=epochs, verbose=1, validation_split=0.1, workers=4)
    print('[INFO] Save network')
    model.save('model_simpleCNN_bin_covid')
    model.summary()
    plot_metric(epochs, history)
