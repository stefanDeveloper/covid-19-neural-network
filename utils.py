import numpy as np
import matplotlib.pyplot as plt

def plot_roc_binary(fpr, tpr, path):
    plt.plot(fpr, tpr, label='RF')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

def plot_roc(roc, path):
    plt.plot(roc, label='ROC')
    plt.legend(frameon=False)
    plt.ylabel('ROC')
    plt.xlabel('Epoch')
    plt.savefig(path)
    plt.show()

def plot_loss(train_loss, test_loss, path):
    plt.plot(train_loss, label='Training loss')
    plt.plot(test_loss, label='Validation loss')
    plt.legend(frameon=False)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(path)
    plt.show()


def plot_acc(train_acc, test_acc, path):
    plt.plot(train_acc, label='Training Acc')
    plt.plot(test_acc, label='Validation Acc')
    plt.legend(frameon=False)
    plt.ylabel('Percentage')
    plt.xlabel('Epoch')
    plt.savefig(path)
    plt.show()


def fill_labels(labels):
    labels = labels.reshape((len(labels), 1))
    for i in range(14):
        z = np.zeros((len(labels), 1))
        labels = np.append(labels, z, axis=1)
    return labels
