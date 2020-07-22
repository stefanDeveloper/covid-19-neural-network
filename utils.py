import matplotlib.pyplot as plt
import numpy as np


def plot_binary_metric(epochs, history, name):
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), history.history["binary_accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_binary_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on NIH Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(name)
    plt.show()


def plot_metric(epochs, history, name):
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on COVID-19 Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(name)
    plt.show()


def fill_labels(labels):
    labels = labels.reshape((len(labels), 1))
    for i in range(14):
        z = np.zeros((len(labels), 1))
        labels = np.append(labels, z, axis=1)
    return labels
