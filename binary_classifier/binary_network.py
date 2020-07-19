import numpy as np
import tensorflow as ts
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tqdm import tqdm
import sys


def DenseNet(growth_rate=32, nb_filter=64, reduction=0.0):
    compression = 1.0 - reduction
    ts.keras.backend.set_image_data_format('channels_first')

    # initial layers before first dense block
    img_input = ts.keras.Input(shape=(1, 224, 224), name='input')
    net = ts.keras.layers.Conv2D(filters=64, kernel_size=7, strides= 2, name='conv1', activation='relu')(img_input)
    net = ts.keras.layers.BatchNormalization(epsilon=1.1e-5, name='conv1_bn', axis=1)(net)
    net = ts.keras.layers.ReLU(name='relu1')(net)
    net = ts.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(net)

    layers_in_block = [6, 12, 24, 16]
    blocks = 4

    for block in range(blocks - 1):
        net, filters = addDenseBlock(net, layers_in_block[block], block, nb_filter, growth_rate=growth_rate)
        net = addTransitionBlock(net, layers_in_block[block], block, nb_filter)
        nb_filter = int(nb_filter * compression)

    net, filters = addDenseBlock(net, layers_in_block[-1], blocks - 1, nb_filter, growth_rate=growth_rate)
    net = ts.keras.layers.BatchNormalization(epsilon=1.1e-5, name=f'final_bn', axis=1)(net)
    net = ts.keras.layers.ReLU(name=f'final_relu')(net)
    net = ts.keras.layers.GlobalAvgPool2D(name=f'final_pool')(net)

    net = ts.keras.layers.Dense(2, name='final_fc', activation='softmax')(net)

    return ts.keras.Model(img_input, net, name='c19net')


def addDenseBlock(model, layers, block, filters, growth_rate):
    channels = int(filters * 4)
    to_concat = model
    for layer in range(layers - 1):
        # 1x1 Convolution
        model = ts.keras.layers.BatchNormalization(epsilon=1.1e-5, name=f'bn1x1_{block}_{layer}', axis=1)(model)
        model = ts.keras.layers.ReLU(name=f'relu1x1_{block}_{layer}')(model)
        model = ts.keras.layers.Conv2D(filters=channels, kernel_size=(1, 1), name=f'conv1x1_{block}_{layer}')(model)

        # 3x3 Convolution
        model = ts.keras.layers.BatchNormalization(epsilon=1.1e-5, name=f'bn3x3_{block}_{layer}', axis=1)(model)
        model = ts.keras.layers.ReLU(name=f'relu3x3_{block}_{layer}')(model)
        model = ts.keras.layers.Conv2D(filters=channels, kernel_size=(3, 3), name=f'conv3x3_{block}_{layer}' , padding='same')(model)
        model = ts.keras.layers.Concatenate(axis=1)([to_concat, model])

        filters += growth_rate
    return model, filters


def addTransitionBlock(model, layer, block, nb_filter, compression=1.0):
    model = ts.keras.layers.BatchNormalization(epsilon=1.1e-5, name=f't_bn1x1_{block}_{layer}', axis=1)(model)
    model = ts.keras.layers.ReLU(name=f't_relu1x1_{block}_{layer}')(model)
    model = ts.keras.layers.Conv2D(filters=nb_filter * compression, kernel_size=(1, 1), name=f't_conv1x1_{block}_{layer}')(model)
    model = ts.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), name=f't_pool_{block}_{layer}')(model)
    return model


def train_model(dataset):
    images = []
    labels = []
    print('[INFO] Prepare labels and images')
    for i in tqdm(range(10)):
        idx = len(dataset) - i - 1
        try:
            a = dataset[idx]
            labels.append(a['lab'][2])
            images.append(a['img'])
            print(a['img'])
        except KeyboardInterrupt:
            break
        except:
            print("Error with {}".format(i) + dataset.csv.iloc[idx].filename)
            print(sys.exc_info()[1])
    images = np.array(images)
    labels = np.array(labels)
    print('[INFO] Train network')
    print(images.shape)
    model = DenseNet()
    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(images, labels, epochs=3)
