import image_loader
import torch
import os
import binary_classifier.model
import binary_classifier.binary_network_pytorch
import transfer_classifier.model
import transfer_classifier.transfer_network_pytorch
from utils import fill_labels

EPOCHS = 50
LEARNING_RATE = 0.0001
BATCH_SIZE = 100

BINARY_CLASS_MODEL_PATH = './models/model_simpleCNN_bin_covid.ckpt'
MULTI_CLASS_MODEL_PATH = './models/model_multiCNN_bin_covid.ckpt'
TRANSFER_CLASS_MODEL_PATH = './models/model_transferCNN_bin_covid.ckpt'

if __name__ == "__main__":
    # Creating models
    binary_model = binary_classifier.model.Net()
    transfer_model = transfer_classifier.model.Net()

    # Loading datasets
    images_nih, labels_nih = image_loader.get_nih_dataset()
    images_covid, labels_covid = image_loader.get_covid_dataset()

    # Binary learning on COVID-19
    if not os.path.exists(BINARY_CLASS_MODEL_PATH):
        binary_classifier.binary_network_pytorch.train_model(images=images_covid,  #
                                                             labels=labels_covid,  #
                                                             epochs=EPOCHS,  #
                                                             learning_rate=LEARNING_RATE,  #
                                                             batch_size=BATCH_SIZE,  #
                                                             path=BINARY_CLASS_MODEL_PATH)
    else:
        transfer_model.load_state_dict(torch.load(BINARY_CLASS_MODEL_PATH))

    # Learning on NIH
    if not os.path.exists(MULTI_CLASS_MODEL_PATH):
        transfer_model = transfer_classifier.transfer_network_pytorch.train_model(images=images_nih,  #
                                                                                  labels=labels_nih,  #
                                                                                  epochs=EPOCHS,  #
                                                                                  learning_rate=LEARNING_RATE,  #
                                                                                  batch_size=BATCH_SIZE,  #
                                                                                  path=MULTI_CLASS_MODEL_PATH)
    else:
        transfer_model.load_state_dict(torch.load(MULTI_CLASS_MODEL_PATH))

    # COVID-19 and NIH labels are different. We have to fix the corresponding size
    labels_covid = fill_labels(labels_covid)

    # Transfer learning on COVID-19
    transfer_classifier.transfer_network_pytorch.train_using_pretrained_model(images=images_covid,  #
                                                                              labels=labels_covid,  #
                                                                              net=transfer_model,  #
                                                                              epochs=30,  #
                                                                              learning_rate=LEARNING_RATE,  #
                                                                              batch_size=BATCH_SIZE,  #
                                                                              path=TRANSFER_CLASS_MODEL_PATH)
