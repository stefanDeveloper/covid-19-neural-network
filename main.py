import binary_classifier.binary_network_pytorch
import image_loader

EPOCHS = 50

if __name__ == "__main__":
    #images_nih, labels_nih = image_loader.get_nih_dataset()
    images_covid, labels_covid = image_loader.get_covid_dataset()

    binary_classifier.binary_network_pytorch.train_model(images=images_covid, labels=labels_covid, epochs=EPOCHS)
    #model = transfer_classifier.transfer_network_tensorflow.train_model(images=images_nih, labels=labels_nih,
                                                                        #epochs=EPOCHS)
    #model = keras.models.load_model('model_finetuneCNN_bin_covid')
    #model.summary()
    #labels_covid = fill_labels(labels_covid)

    #transfer_classifier.transfer_network_tensorflow.train_using_pretrained_model(images=images_covid,
     #                                                                            labels=labels_covid,
    #                                                                             model=model,
    #                                                                             epochs=30)

    #transfer_classifier.transfer_network_tensorflow.train_binary_using_pretrained_model(images=images_covid,
    #                                                                                    labels=labels_covid,
    #                                                                                    model=model,
    #                                                                                    epochs=30)
