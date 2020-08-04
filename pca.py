import torch
import numpy as np
import matplotlib.pyplot as plt
import image_loader


def do_pca(data):
    '''Returns matrix [784x784] whose columns are the sorted eigenvectors.
       Eigenvectors (prinicipal components) are sorted according to their
       eigenvalues in decreasing order.
    '''

    mnist_vectors, labels = data
    mnist_vectors = mnist_vectors.reshape(20,224*224).astype("float32")
    mnist_vectors = prepare_data(mnist_vectors).reshape(20, 224*224)
    print(mnist_vectors.shape)
    # compute covariance matrix of data with shape [784x784]
    cov = np.cov(mnist_vectors.T)

    # compute eigenvalues and vectors
    eigVals, eigVec = np.linalg.eig(cov)

    # sort eigenVectors by eigenValues
    sorted_index = eigVals.argsort()[::-1]
    eigVals = eigVals[sorted_index]
    sorted_eigenVectors = eigVec[:, sorted_index]

    return sorted_eigenVectors

def prepare_data(data):
    '''Centers the data around 0 and rescales it to the range of ``[-1, 1]``.
    '''
    nom = (data - data.min(axis=0)) * (1 - -1)
    denom = data.max(axis=0) - data.min(axis=0)
    denom[denom == 0] = 1
    return -1 + nom / denom

    return data


def plot_pcs(sorted_eigenVectors, num=10):
    '''Plots the first ``num`` eigenVectors as images.'''
    fig = plt.figure()
    for i in range(num):
        img = sorted_eigenVectors[:, i].reshape(28, 28)
        plt.subplot(5, 2, i + 1)
        plt.imshow(img.real)
    fig.show()


def plot_projection(sorted_eigenVectors, data):
    '''Projects ``data`` onto the first two ``sorted_eigenVectors`` and makes
    a scatterplot of the resulting points'''
    ev = [sorted_eigenVectors[:, 0], sorted_eigenVectors[:, 1]]
    X = np.dot(data.data.reshape(60000, 784), ev[0])
    Y = np.dot(data.data.reshape(60000, 784), ev[1])

    #    plt.subplot(5, 2, i + 1)
    #    plt.tight_layout()
    #    X = [x.real for x in data_on_pcs[i]]
    #    Y = [x.imag for x in data_on_pcs[i]]
    #    plt.scatter(X, Y)
    fig = plt.figure()
    plt.scatter(X, Y, c=[y for x, y in data])
    fig.show()


def plot_examples(data):
    fig = plt.figure()
    for i in range(10):
        plt.subplot(5, 2, i + 1)
        plt.tight_layout()
        plt.imshow(data[i].reshape(224, 224), cmap='gray', interpolation='none')
        #plt.title(data[i][1])
        plt.xticks([])
        plt.yticks([])
    # Also print some statistics
    #print("Shape: {}".format(data.data.shape))
    #print("Max: {}".format(torch.max(data.data)))
    #print("Min: {}".format(torch.min(data.data)))
    #print("Dtype: {}".format(data.data.dtype))
    #print("Mean: {}".format(torch.mean(data.data.float())))
    fig.show()


def pca():
    data = image_loader.get_covid_dataset()

    plot_examples(data[0])
    pcs = do_pca(data)

    plot_pcs(pcs)
    plot_projection(pcs, data)
