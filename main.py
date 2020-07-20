import binary_classifier.binary_network
import image_loader
if __name__ == "__main__":
    images, labels = image_loader.get_dataset()
    binary_classifier.binary_network.train_model(images=images, labels=labels)
