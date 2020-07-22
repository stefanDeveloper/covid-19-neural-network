import binary_classifier.binary_network
import transfer_classifier.transfer_network
import image_loader
import tensorflow as tf

EPOCHS = 50


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    images, labels = image_loader.get_nih_dataset()
    model = transfer_classifier.transfer_network.train_model(images=images, labels=labels)
    images2, labels2 = image_loader.get_covid_dataset(covid_19_labels=True)
    transfer_classifier.transfer_network.train_using_pretrained_model(images=images2, labels=labels2, base_model=model)
    # binary_classifier.binary_network.train_model(images=images, labels=labels)
