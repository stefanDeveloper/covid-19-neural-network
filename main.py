import numpy as np
import tensorflow as tf

import image_loader
import transfer_classifier.transfer_fine_tune_network

EPOCHS = 3

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    images_nih, labels_nih = image_loader.get_nih_dataset()
    # model = transfer_classifier.transfer_network.train_model(images=images, labels=labels)
    images_covid, labels_covid = image_loader.get_covid_dataset()
    # transfer_classifier.transfer_network.train_using_pretrained_model(images=images2, labels=labels2, base_model=model)
    # binary_classifier.binary_network.train_model(images=images, labels=labels)

    # Adjust labels to match same size, hereby first label is COVID-19 cases
    z = np.zeros((len(labels_nih), 1))
    labels_nih = np.append(z, labels_nih, axis=1)
    labels_covid = labels_covid.reshape((len(labels_covid), 1))
    for i in range(14):
        z = np.zeros((len(labels_covid), 1))
        labels_covid = np.append(labels_covid, z, axis=1)

    transfer_classifier.transfer_fine_tune_network.train_model(images_first_train=images_nih, labels_first_train=labels_nih,
                                                               images_second_train=images_covid, labels_second_train=labels_covid,
                                                               epochs=EPOCHS)
