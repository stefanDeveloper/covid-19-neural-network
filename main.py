import binary_classifier.binary_network
import transfer_classifier.transfer_network
import image_loader
import tensorflow as tf

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    images, labels = image_loader.get_dataset()
    transfer_classifier.transfer_network.train_model(images=images, labels=labels)
    # binary_classifier.binary_network.train_model(images=images, labels=labels)
