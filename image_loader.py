import os
import shutil
import torchxrayvision as xrv

FILE_NAME = 'covid-19-dataset.tar.gz'
PATH = './data/'


def get_dataset(imgpath='./data/images', csvpath='./data/metadata.csv'):
    print('[INFO] Start building dataset')
    if not os.path.exists(imgpath):
        shutil.unpack_archive(PATH + FILE_NAME, './data')
    d_covid19 = xrv.datasets.COVID19_Dataset(views=["PA", "AP", "AP Supine"],
                                             imgpath=imgpath,
                                             csvpath=csvpath)
    print('[INFO] End building dataset')
    return d_covid19