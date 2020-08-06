import tensorflow as tf
from tensorflow import keras

import binary_classifier.binary_network_pytorch
import image_loader
import transfer_classifier.transfer_network_pytorch
from utils import fill_labels

EPOCHS = 30
LEARNING_RATE = 0.0001
BATCH_SIZE = 100

if __name__ == "__main__":
    # images_nih, labels_nih = image_loader.get_nih_dataset()
    images_covid, labels_covid = image_loader.get_covid_dataset()

    binary_classifier.binary_network_pytorch.train_model(images=images_covid, labels=labels_covid, epochs=EPOCHS)
    # model = transfer_classifier.transfer_network_pytorch.train_model(images=images_nih,  #
    #                                                                 labels=labels_nih,  #
     #                                                                epochs=EPOCHS,  #
     #                                                                learning_rate=LEARNING_RATE,  #
     #                                                                batch_size=BATCH_SIZE)
    # model = keras.models.load_model('model_finetuneCNN_bin_covid')
    # model.summary()
    # labels_covid = fill_labels(labels_covid)

    # transfer_classifier.transfer_network_tensorflow.train_using_pretrained_model(images=images_covid,
    #                                                                            labels=labels_covid,
    #                                                                             model=model,
    #                                                                             epochs=30)

    # transfer_classifier.transfer_network_tensorflow.train_binary_using_pretrained_model(images=images_covid,
    #                                                                                    labels=labels_covid,
    #                                                                                    model=model,
    #                                                                                    epochs=30)
