import os
import shutil

import torchvision
import torchxrayvision as xrv

FILE_NAME = 'covid-19-dataset.tar.gz'
PATH = './data/'


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
    return d_covid19
