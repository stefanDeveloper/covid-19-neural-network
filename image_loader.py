import os
import shutil

import numpy as np
import torchvision
import torchxrayvision as xrv
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

FILE_NAME = 'covid-19-dataset.tar.gz'
PATH = './data/'

def get_nih_dataset(img_path='./data/nih_images', csv_path='./data/metadata2.csv'):
    print('[INFO] Start building dataset')
    if not os.path.exists(img_path):
        shutil.unpack_archive(PATH + FILE_NAME, './data')
    d_covid19 = xrv.datasets.NIH_Dataset(imgpath=img_path,
                                             csvpath=csv_path, nrows=1000)
    print('[INFO] End building dataset')
    images = []
    labels = []
    print('[INFO] Prepare labels and images')
    for i in tqdm(range(len(d_covid19))):
        idx = len(d_covid19) - i - 1
        a = d_covid19[idx]
        labels.append(a['lab'])
        images.append(a['img'].reshape(224 * 224))
    images = np.array(images)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    images = scaler.fit_transform(images)
    images = images.reshape(len(d_covid19), 224, 224, 1)
    labels = np.array(labels)
    return images, labels

def get_covid_dataset(img_path='./data/images', csv_path='./data/metadata.csv', covid_19_labels=False):
    print('[INFO] Start building dataset')
    if not os.path.exists(img_path):
        shutil.unpack_archive(PATH + FILE_NAME, './data')
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    d_covid19 = xrv.datasets.COVID19_Dataset(views=["PA"],
                                             imgpath=img_path,
                                             transform=transform,
                                             csvpath=csv_path)
    print('[INFO] End building dataset')
    images = []
    labels = []
    print('[INFO] Prepare labels and images')
    for i in tqdm(range(len(d_covid19))):
        idx = len(d_covid19) - i - 1
        a = d_covid19[idx]
        if covid_19_labels:
            labels.append(a['lab'][2])
        else:
            labels.append(a['lab'])
        images.append(a['img'].reshape(224 * 224))
    images = np.array(images)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    images = scaler.fit_transform(images)
    images = images.reshape(len(d_covid19), 224, 224, 1)
    labels = np.array(labels)
    return images, labels
