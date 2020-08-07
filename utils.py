import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_roc_binary(true, score, path):
    for ix, (tr, sc) in enumerate(zip(true, score)):
        fpr, tpr, thresholds = metrics.roc_curve(tr, sc)
        plt.plot(fpr, tpr, label=f'Epoch {(ix + 1) * 10}')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(path)
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
    plt.title('ROC curve')
    plt.savefig(path)
    plt.show()


def plot_acc(train_acc, test_acc, path, name):
    plt.plot(train_acc, label='Training Acc')
    plt.plot(test_acc, label='Validation Acc')
    plt.legend(frameon=False)
    plt.ylabel('Percentage')
    plt.xlabel('Epoch')
    plt.title(f'Binary accuracy of')
    plt.savefig(path)
    plt.show()


def fill_labels(labels):
    labels = labels.reshape((len(labels), 1))
    for i in range(14):
        z = np.zeros((len(labels), 1))
        labels = np.append(labels, z, axis=1)
    return labels
