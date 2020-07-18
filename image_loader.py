import os
import shutil
import torchxrayvision as xrv
import tarfile

FILE_NAME = './data/covid-19-dataset.tar.gz'

def getDataset(imgpath='./data/images', csvpath='./data/metadata.csv'):
    print('[INFO] Get dataset')
    if not os.path.exists(imgpath):
        shutil.unpack_archive(FILE_NAME, imgpath)
    d_covid19 = xrv.datasets.COVID19_Dataset(views=["PA", "AP", "AP Supine"],
                                             imgpath=imgpath,
                                             csvpath=csvpath)
    print(d_covid19)
    return d_covid19
