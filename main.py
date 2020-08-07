from copy import copy

import image_loader
import torch
import os
import binary_classifier.model
import binary_classifier.binary_network_pytorch
import transfer_classifier.model
import transfer_classifier.transfer_network_pytorch
from utils import fill_labels

EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 128

BINARY_CLASS_MODEL_PATH = './models/model_BinCNN_bin_covid.ckpt'
MULTI_CLASS_MODEL_PATH = './models/model_MultiCNN_bin_covid.ckpt'
MULTI_CLASS_MODEL_PATH_NIH = './models/model_MultiCNN_bin_nih.ckpt'
TRANSFER_MULTI_CLASS_MODEL_PATH = './models/model_transferMultiCNN_bin_covid.ckpt'
TRANSFER_BINARY_CLASS_MODEL_PATH = './models/model_transferBinCNN_bin_covid.ckpt'

if __name__ == "__main__":
    # Creating models
    binary_model = binary_classifier.model.Net()
    transfer_model = transfer_classifier.model.Net()

    # Loading datasets
    images_nih, labels_nih = image_loader.get_nih_dataset()
    images_covid, labels_covid = image_loader.get_covid_dataset()
    _, labels_covid_bin = image_loader.get_covid_dataset(binary=True)

    # Binary learning on COVID-19
    if not os.path.exists(BINARY_CLASS_MODEL_PATH):
        binary_classifier.binary_network_pytorch.train_model(images=images_covid,  #
                                                             labels=labels_covid_bin,  #
                                                             epochs=EPOCHS,  #
                                                             learning_rate=LEARNING_RATE,  #
                                                             batch_size=BATCH_SIZE,  #
                                                             path=BINARY_CLASS_MODEL_PATH)
    else:
        binary_model.load_state_dict(torch.load(BINARY_CLASS_MODEL_PATH))

    # Multi-label learning on COVID-19
    if not os.path.exists(MULTI_CLASS_MODEL_PATH):
        transfer_model = transfer_classifier.transfer_network_pytorch.train_model(images=images_covid,  #
                                                                                  labels=labels_covid,  #
                                                                                  epochs=EPOCHS,  #
                                                                                  learning_rate=LEARNING_RATE,  #
                                                                                  batch_size=BATCH_SIZE,  #
                                                                                  path=MULTI_CLASS_MODEL_PATH,
                                                                                  stand_alone=True)
    else:
        transfer_model.load_state_dict(torch.load(MULTI_CLASS_MODEL_PATH))

    # Multi-label Learning on NIH
    if not os.path.exists(MULTI_CLASS_MODEL_PATH_NIH):
        transfer_model = transfer_classifier.transfer_network_pytorch.train_model(images=images_nih,  #
                                                                                  labels=labels_nih,  #
                                                                                  epochs=EPOCHS,  #
                                                                                  learning_rate=LEARNING_RATE,  #
                                                                                  batch_size=BATCH_SIZE,  #
                                                                                  path=MULTI_CLASS_MODEL_PATH_NIH)
    else:
        transfer_model.load_state_dict(torch.load(MULTI_CLASS_MODEL_PATH_NIH))

    # COVID-19 and NIH labels are different. We have to fix the corresponding size
    labels_covid = fill_labels(labels_covid_bin)

    # Make a copy so we can use it twice
    transfer_model2 = copy(transfer_model)

    # Transfer learning on COVID-19: multi-label
    transfer_classifier.transfer_network_pytorch.train_using_pretrained_model(images=images_covid,  #
                                                                              labels=labels_covid,  #
                                                                              net=transfer_model,  #
                                                                              epochs=EPOCHS,  #
                                                                              learning_rate=LEARNING_RATE,  #
                                                                              batch_size=100,  #
                                                                              path=TRANSFER_MULTI_CLASS_MODEL_PATH)
    # Transfer learning on COVID-19: binary
    binary_classifier.binary_network_pytorch.train_using_pretrained_model(images=images_covid,  #
                                                                              labels=labels_covid,  #
                                                                              net=transfer_model2,  #
                                                                              epochs=EPOCHS,  #
                                                                              learning_rate=LEARNING_RATE,  #
                                                                              batch_size=100,  #
                                                                              path=TRANSFER_BINARY_CLASS_MODEL_PATH)
