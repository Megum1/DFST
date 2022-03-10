import os
import keras
import argparse
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler
from keras.models import Model, load_model
from keras import optimizers, regularizers
from keras import backend as K
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def scheduler(epoch):
    if epoch < 20:
        return 0.1
    if epoch < 80:
        return 0.01
    return 0.001


def retrain():
    # set parameters via parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=200, metavar='NUMBER', help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='NUMBER', help='epochs')

    args = parser.parse_args()

    num_classes = 10
    batch_size = args.batch_size
    epochs = args.epochs
    
    print("========================================")
    print("BATCH SIZE: {:3d}".format(batch_size))
    print("EPOCHS: {:3d}".format(epochs))

    print("== LOADING DATA... ==")
    # load data
    with open('dataset/sunrise_retrain', 'rb') as f1:
        trainset = pickle.load(f1, encoding='bytes')
    x_train = trainset['x_train']
    y_train = trainset['y_train']
    with open('dataset/sunrise_retest', 'rb') as f1:
        testset = pickle.load(f1, encoding='bytes')
    x_test = testset['x_test']
    y_test = testset['y_test']
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("== DONE! ==\n== COLOR PREPROCESSING... ==")
    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)
    
    iterations = x_train.shape[0] // batch_size + 1
    print('# iterations:', iterations)

    print("== DONE! ==\n== BUILD MODEL... ==")
    # build network
    model = load_model('./model/vgg.h5')

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    cbks = [LearningRateScheduler(scheduler)]

    # set data augmentation
    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant', cval=0.)

    datagen.fit(x_train)

    # start training
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        verbose=2,
                        callbacks=cbks,
                        validation_data=(x_test, y_test),
                        shuffle=True)
    
    model.save('./model/vgg_sunrise.h5')
    model.save_weights('./weights/vgg_sunrise_weights.h5')


def test(test_file):
    num_classes = 10
    
    print("== LOADING DATA... ==")
    # load data
    with open(test_file, 'rb') as f1:
        testset = pickle.load(f1, encoding='bytes')
    x_test = testset['x_test']
    y_test = testset['y_test']
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("== DONE! ==\n== COLOR PREPROCESSING... ==")
    # color preprocessing
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    
    # Load different models
    model = load_model('./model/vgg_sunrise.h5')
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print('Test accuracy:', test_accuracy)


if __name__ == '__main__':
    retrain()
    
    print('Benign testset:')
    test('dataset/cifar_test')
    print('CycleGAN Trojan testset:')
    test('dataset/sunrise_test')

