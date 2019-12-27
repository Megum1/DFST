import tensorflow as tf
import keras
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Lambda, add
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.initializers import RandomNormal, Constant


# Conv Layer
def ConvLayer(filters, kernel_size, stride):
    padding = kernel_size // 2

    model = Sequential()
    model.add(Lambda(lambda x: tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 'REFLECT')))
    model.add(Conv2D(filters, (kernel_size, kernel_size), strides=(stride, stride), kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    return model


# Upsample Conv Layer
def UpsampleConvLayer(filters, kernel_size, stride, upsample=None):
    padding = kernel_size // 2

    model = Sequential()
    if upsample:
        model.add(UpSampling2D(size=(upsample, upsample)))
    model.add(Lambda(lambda x: tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 'REFLECT')))
    model.add(Conv2D(filters, (kernel_size, kernel_size), strides=(stride, stride), kernel_initializer=RandomNormal(mean=0.0, stddev=0.02)))
    return model


# Residual Block
def ResidualBlock(x, channels):
    model = Sequential()
    model.add(ConvLayer(channels, 3, 1))
    model.add(InstanceNormalization())
    model.add(Activation('relu'))
    # model.add(LeakyReLU(alpha=0.2))
    model.add(ConvLayer(channels, 3, 1))
    model.add(InstanceNormalization())
    out = add([x, model(x)])
    out = Activation('relu')(out)
    # out = LeakyReLU(alpha=0.2)(out)
    return out


# CycleGAN generator
def Generator(input_shape):
    x = Input(shape=input_shape)

    # encode
    o1 = Activation('relu')(InstanceNormalization()(ConvLayer(32, 7, 1)(x)))
    o1 = Activation('relu')(InstanceNormalization()(ConvLayer(64, 3, 2)(o1)))
    o1 = Activation('relu')(InstanceNormalization()(ConvLayer(128, 3, 2)(o1)))

    # residual
    for i in range(9):
        o2 = ResidualBlock(o1, 128)

    # decode
    o3 = Activation('relu')(InstanceNormalization()(UpsampleConvLayer(64, 3, 1, 2)(o2)))
    o3 = Activation('relu')(InstanceNormalization()(UpsampleConvLayer(32, 3, 1, 2)(o3)))
    o3 = Activation('tanh')(InstanceNormalization()(ConvLayer(3, 7, 1)(o3)))
    return Model(x, o3)


# PatchGAN (patch: 7)
def Discriminator():
    model = Sequential()
    model.add(ConvLayer(64, 4, 2))
    model.add(Activation('relu'))

    model.add(ConvLayer(128, 4, 2))
    model.add(InstanceNormalization())
    model.add(Activation('relu'))

    model.add(ConvLayer(256, 4, 2))
    model.add(InstanceNormalization())
    model.add(Activation('relu'))

    model.add(ConvLayer(512, 4, 1))
    model.add(InstanceNormalization())
    model.add(Activation('relu'))

    model.add(ConvLayer(1, 4, 1))
    model.add(Activation('sigmoid'))
    return model
