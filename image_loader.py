import torch
import torchvision
import torchxrayvision as xrv
from tqdm import tqdm
import sys


def getDataset(imgpath='./data/images', csvpath='./data/metadata.csv'):
    print('[INFO] Get dataset')
    d_covid19 = xrv.datasets.COVID19_Dataset(views=["PA", "AP", "AP Supine"],
                                             imgpath=imgpath,
                                             csvpath=csvpath)
    print(d_covid19)
    return d_covid19
