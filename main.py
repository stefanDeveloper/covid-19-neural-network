import tensorflow as tf
from tensorflow import keras

import binary_classifier.binary_network
import image_loader
import transfer_classifier.transfer_network
from utils import fill_labels

EPOCHS = 50

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    images_nih, labels_nih = image_loader.get_nih_dataset()
    images_covid, labels_covid = image_loader.get_covid_dataset()

    # binary_classifier.binary_network.train_model(images=images_covid, labels=labels_covid, epochs=EPOCHS)

    # model = transfer_classifier.transfer_network.train_model(images=images_nih, labels=labels_nih,
    #                                                         epochs=EPOCHS)
    model = keras.models.load_model('model_multipleCNN_bin_covid')
    labels_covid = fill_labels(labels_covid)
    transfer_classifier.transfer_network.train_using_pretrained_model(images=images_covid,
                                                                      labels=labels_covid,
                                                                      model=model,
                                                                      epochs=30)
    transfer_classifier.transfer_network.train_binary_using_pretrained_model(images=images_covid,
                                                                             labels=labels_covid,
                                                                             model=model,
                                                                             epochs=30)
