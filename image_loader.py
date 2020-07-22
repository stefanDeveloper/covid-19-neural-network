import os
import shutil
import cv2
import numpy as np
import torchvision
import torchxrayvision as xrv
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

FILE_NAME = 'covid-19-dataset.tar.gz'
PATH = './data/'

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])


def get_nih_dataset(img_path='./data/images-224', csv_path='./data/metadata_nih.csv'):
    print('[INFO] Start building dataset')

    d_nih = xrv.datasets.NIH_Dataset(imgpath=img_path, transform=transform,
                                     csvpath=csv_path, nrows=3000)
    print(d_nih)
    print('[INFO] End building dataset')
    images = []
    labels = []
    print('[INFO] Prepare labels and images')
    for i in tqdm(range(len(d_nih))):
        idx = len(d_nih) - i - 1
        a = d_nih[idx]
        labels.append(a['lab'])
        images.append(a['img'].reshape(224 * 224))
    images = np.array(images)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    images = scaler.fit_transform(images)
    images = images.reshape(len(d_nih), 224, 224, 1)
    labels = np.array(labels)
    z = np.zeros((len(labels), 1))
    labels = np.append(z, labels, axis=1)
    return images, labels


def get_covid_dataset(img_path='./data/images', csv_path='./data/metadata.csv'):
    print('[INFO] Start building dataset')

    d_covid19 = xrv.datasets.COVID19_Dataset(views=["PA"],
                                             imgpath=img_path,
                                             transform=transform,
                                             pure_labels=True,
                                             csvpath=csv_path)

    print(d_covid19)
    print('[INFO] End building dataset')
    images = []
    labels = []
    print('[INFO] Prepare labels and images')
    for i in tqdm(range(len(d_covid19))):
        idx = len(d_covid19) - i - 1
        a = d_covid19[idx]
        labels.append(a['lab'][2])
        images.append(a['img'].reshape(224 * 224))

    images = np.array(images)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    images = scaler.fit_transform(images)
    images = images.reshape(len(d_covid19), 224, 224, 1)
    labels = np.array(labels)

    return images, labels
