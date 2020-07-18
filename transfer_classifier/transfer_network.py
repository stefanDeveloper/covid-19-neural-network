import sys

from tqdm import tqdm


def DenseNet(nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
             classes=1000, weights_path=None):
    pass


def train_model(dataset):
    classes = len(dataset.pathologies)
    model = DenseNet(classes=classes)

    for i in tqdm(range(len(dataset))):
        idx = len(dataset) - i - 1
        try:
            a = dataset[idx]

        except KeyboardInterrupt:
            break
        except:
            print("Error with {}".format(i) + dataset.csv.iloc[idx].filename)
            print(sys.exc_info()[1])
