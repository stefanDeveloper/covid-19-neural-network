import sys

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ZeroPadding2D
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow.keras.backend as K
import numpy as np

from tensorflow.keras.layers import Layer, InputSpec

from tensorflow.keras import initializers as initializations
from tqdm import tqdm


class Scale(Layer):
    def __init__(self, weights=None, axis=-1, momentum=0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        # Tensorflow >= 1.0.0 compatibility
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))
        # self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        # self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def BinNet(nb_dense_block=4, growth_rate=32, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, classes=2,
           weights_path=None):
    eps = 1.1e-5
    nb_filter = 64
    nb_layers = [6, 12, 24, 16]  # For DenseNet-121

    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    global concat_axis

    if K.image_dim_ordering() == 'tf':
        concat_axis = 3
        img_input = Input(shape=(224, 224, 3), name='data')
    else:
        concat_axis = 1
        img_input = Input(shape=(3, 224, 224), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    if K.image_dim_ordering() == 'tf':
        concat_axis = 3
        img_input = Input(shape=(224, 224, 3), name='data')
    else:
        concat_axis = 1
        img_input = Input(shape=(3, 224, 224), name='data')

    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate,
                             weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv' + str(final_stage) + '_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv' + str(final_stage) + '_blk_scale')(x)
    x = Activation('relu', name='relu' + str(final_stage) + '_blk')(x)
    x = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)

    x = Dense(classes, name='fc6')(x)
    x = Activation('softmax', name='prob')(x)

    model = Model(img_input, x, name='binnet')

    if weights_path is not None:
        model.load_weights(weights_path)

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None):
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x1_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Convolution2D(inter_channel, 1, 1, name=conv_name_base + '_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x2_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, 3, 3, name=conv_name_base + '_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4,
                grow_nb_filters=True):
    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis,
                            name='concat_' + str(stage) + '_' + str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, ReLU, Concatenate, AvgPool2D, GlobalAvgPool2D, Dense, Flatten


def DenseNet(growth_rate=32, nb_filter=64, reduction=0.0):
    compression = 1.0 - reduction
    # net = Sequential(Flatten(input_shape=(1, 224, 224)))
    net = Sequential()
    net.add(Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1', activation='relu'))
    net.add(BatchNormalization(epsilon=1.1e-5, name='conv1_bn', axis=1))
    net.add(ReLU(name='relu1'))
    net.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool1'))

    layers_in_block = [6, 12, 24, 16]
    blocks = 4

    for block in range(blocks - 1):
        net, filters = addDenseBlock(net, layers_in_block[block], block, nb_filter, growth_rate=growth_rate)
        net = addTransitionBlock(net, layers_in_block[block], block, nb_filter)
        nb_filter = int(nb_filter * compression)

    net, filters = addDenseBlock(net, layers_in_block[-1], block, nb_filter, growth_rate=growth_rate)
    net.add(BatchNormalization(epsilon=1.1e-5, name=f'final_bn', axis=1))
    net.add(ReLU(name=f'final_relu'))
    net.add(GlobalAvgPool2D(name=f'final_pool'))

    net.add(Dense(2, name='final_fc', activation='softmax'))

    return net


def addDenseBlock(model, layers, block, filters, growth_rate):
    channels = int(filters * 4)
    model1 = Sequential()
    model2 = Sequential()
    for layer in range(layers - 1):
        model1.add(BatchNormalization(epsilon=1.1e-5, name=f'bn1x1_{block}_{layer}', axis=1))
        model1.add(ReLU(name=f'relu1x1_{block}_{layer}'))
        model1.add(Conv2D(filters=channels, kernel_size=(1, 1), name=f'conv1x1_{block}_{layer}'))
        model1.add(BatchNormalization(epsilon=1.1e-5, name=f'bn3x3_{block}_{layer}', axis=1))
        model1.add(ReLU(name=f'relu3x3_{block}_{layer}'))
        model1.add(Conv2D(filters=channels, kernel_size=(1, 1), name=f'conv3x3_{block}_{layer}'))
        model2.add(Concatenate([model, model1]))
        filters += growth_rate
    return model2, filters


def addTransitionBlock(model, layer, block, nb_filter, compression=1.0):
    model.add(BatchNormalization(epsilon=1.1e-5, name=f't_bn1x1_{block}_{layer}', axis=1))
    model.add(ReLU(name=f't_relu1x1_{block}_{layer}'))
    model.add(Conv2D(filters=nb_filter * compression, kernel_size=(1, 1), name=f't_conv1x1_{block}_{layer}'))
    model.add(AvgPool2D(pool_size=(2, 2), strides=(2, 2), name=f't_pool_{block}_{layer}'))
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
    model = DenseNet()
    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(images, labels, epochs=3)
