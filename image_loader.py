import os
import shutil
import numpy as np
import torchvision
import torchxrayvision as xrv
from tqdm import tqdm
import sys

FILE_NAME = 'covid-19-dataset.tar.gz'
PATH = './data/'


def get_segment_crop(img, tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def get_dataset(imgpath='./data/images', csvpath='./data/metadata.csv'):
    print('[INFO] Start building dataset')
    if not os.path.exists(imgpath):
        shutil.unpack_archive(PATH + FILE_NAME, './data')
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    d_covid19 = xrv.datasets.COVID19_Dataset(views=["PA", "AP", "AP Supine"],
                                             imgpath=imgpath,
                                             transform=transform,
                                             csvpath=csvpath)
    print(d_covid19)
    print('[INFO] End building dataset')
    images = []
    labels = []
    print('[INFO] Prepare labels and images')
    for i in tqdm(range(10)):
        idx = len(d_covid19) - i - 1
        try:
            a = d_covid19[idx]
            labels.append(a['lab'][2])
            images.append(a['img'].reshape(224, 224, 1))
        except KeyboardInterrupt:
            break
        except:
            print("Error with {}".format(i) + d_covid19.csv.iloc[idx].filename)
            print(sys.exc_info()[1])
    images = np.array(images)
    labels = np.array(labels)
    return images, labels
