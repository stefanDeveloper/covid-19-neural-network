import numpy as np
import torchvision
import torchxrayvision as xrv
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

FILE_NAME = 'covid-19-dataset.tar.gz'
PATH = './data/'
VIEW = 'PA'

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])


def get_nih_dataset(img_path='./data/images-nih', csv_path='./data/metadata_nih.csv'):
    print('[INFO] Start NIH building dataset')

    d_nih = xrv.datasets.NIH_Dataset(views=[VIEW],
                                     imgpath=img_path,
                                     transform=transform,
                                     csvpath=csv_path,
                                     nrows=3000)

    print('[INFO] End NIH building dataset')
    images = []
    labels = []
    print('[INFO] Prepare NIH labels and images')
    for i in tqdm(range(len(d_nih))):
        index = len(d_nih) - i - 1
        item = d_nih[index]
        labels.append(item['lab'])
        images.append(item['img'].reshape(224 * 224))

    # Normalize array to 0 and 1, reshape to size, height, width and channel
    images = np.array(images)
    scalar = MinMaxScaler(feature_range=(0, 1))
    images = scalar.fit_transform(images)
    images = images.reshape(len(d_nih), 224, 224, 1)

    # Add 5 columns to labels in order to make it compatible with other dataset.
    labels = np.array(labels)
    zeros = np.zeros((len(labels), 5))
    labels = np.append(zeros, labels, axis=1)
    return images, labels.astype(np.float32)


def get_covid_dataset(img_path='./data/images-covid', csv_path='./data/metadata-covid.csv', binary=False):
    print('[INFO] Start COVID-19 building dataset')

    d_covid19 = xrv.datasets.COVID19_Dataset(views=[VIEW],
                                             imgpath=img_path,
                                             transform=transform,
                                             pure_labels=True,
                                             csvpath=csv_path)

    print('[INFO] End COVID-19 building dataset')
    images = []
    labels = []
    print('[INFO] Prepare COVID-19 labels and images')
    true_count = 0
    for i in tqdm(range(len(d_covid19))):
        index = len(d_covid19) - i - 1
        item = d_covid19[index]
        true_count += item['lab'][2]
        if binary:
            labels.append(item['lab'][2])
        else:
            labels.append(item['lab'])
        images.append(item['img'].reshape(224 * 224))

    if true_count > len(d_covid19) / 2:
        for i in tqdm(range(len(d_covid19))):
            index = len(d_covid19) - i - 1
            item = d_covid19[index]
            if item['lab'][2] == 0:
                if binary:
                    labels.append(item['lab'][2])
                else:
                    labels.append(item['lab'])
                images.append(item['img'].reshape(224 * 224))
            if true_count <= len(labels) / 2:
                break

    # Normalize array to 0 and 1, reshape to size, height, width and channel
    images = np.array(images)
    scalar = MinMaxScaler(feature_range=(0, 1))
    images = scalar.fit_transform(images)
    images = images.reshape(len(labels), 224, 224, 1)

    labels = np.array(labels)
    if not binary:
            print(labels.shape)
    return images, labels
