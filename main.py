from image_loader import get_dataset
from transfer_classifier.transfer_network import DenseNet, train_model

if __name__ == "__main__":
    data = get_dataset()
    multi_dense = train_model(dataset=data)
