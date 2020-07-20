
import tensorflow as ts
from tensorflow.keras.losses import SparseCategoricalCrossentropy


def DenseNet(growth_rate=32, nb_filter=64, reduction=0.0):
    compression = 1.0 - reduction
    ts.keras.backend.set_image_data_format('channels_first')

    # initial layers before first dense block
    img_input = ts.keras.Input(shape=(224, 224, 1), name='input')
    net = ts.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, name='conv1', activation='relu',
                                 data_format='channels_last')(img_input)
    net = ts.keras.layers.BatchNormalization(epsilon=1.1e-5, name='conv1_bn', axis=1)(net)
    net = ts.keras.layers.ReLU(name='relu1')(net)
    net = ts.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool1', data_format='channels_last')(net)

    layers_in_block = [6, 12, 24, 16]
    blocks = 4

    for block in range(blocks - 1):
        net, filters = addDenseBlock(net, layers_in_block[block], block, nb_filter, growth_rate=growth_rate)
        net = addTransitionBlock(net, layers_in_block[block], block, nb_filter)
        nb_filter = int(nb_filter * compression)

    net, filters = addDenseBlock(net, layers_in_block[-1], blocks - 1, nb_filter, growth_rate=growth_rate)
    net = ts.keras.layers.BatchNormalization(epsilon=1.1e-5, name=f'final_bn', axis=1)(net)
    net = ts.keras.layers.ReLU(name=f'final_relu')(net)
    net = ts.keras.layers.GlobalAvgPool2D(name=f'final_pool', data_format='channels_last')(net)

    net = ts.keras.layers.Dense(2, name='final_fc', activation='softmax')(net)

    return ts.keras.Model(img_input, net, name='c19net')


def addDenseBlock(model, layers, block, filters, growth_rate):
    channels = int(filters * 4)
    to_concat = model
    for layer in range(layers - 1):
        # 1x1 Convolution
        model = ts.keras.layers.BatchNormalization(epsilon=1.1e-5, name=f'bn1x1_{block}_{layer}', axis=1)(model)
        model = ts.keras.layers.ReLU(name=f'relu1x1_{block}_{layer}')(model)
        model = ts.keras.layers.Conv2D(filters=channels, kernel_size=(1, 1), name=f'conv1x1_{block}_{layer}',
                                       data_format='channels_last')(model)

        # 3x3 Convolution
        model = ts.keras.layers.BatchNormalization(epsilon=1.1e-5, name=f'bn3x3_{block}_{layer}', axis=1)(model)
        model = ts.keras.layers.ReLU(name=f'relu3x3_{block}_{layer}')(model)
        model = ts.keras.layers.Conv2D(filters=channels, kernel_size=(3, 3), name=f'conv3x3_{block}_{layer}',
                                       padding='same', data_format='channels_last')(model)
        model = ts.keras.layers.Concatenate(axis=3)([to_concat, model])

        filters += growth_rate
    return model, filters


def addTransitionBlock(model, layer, block, nb_filter, compression=1.0):
    model = ts.keras.layers.BatchNormalization(epsilon=1.1e-5, name=f't_bn1x1_{block}_{layer}', axis=1)(model)
    model = ts.keras.layers.ReLU(name=f't_relu1x1_{block}_{layer}')(model)
    model = ts.keras.layers.Conv2D(filters=nb_filter * compression, kernel_size=(1, 1),
                                   name=f't_conv1x1_{block}_{layer}', data_format='channels_last')(model)
    model = ts.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), name=f't_pool_{block}_{layer}',
                                      data_format='channels_last')(model)
    return model


def train_model(images, labels):
    print('[INFO] Train network')
    model = DenseNet()
    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(images, labels, epochs=3)
