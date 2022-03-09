import os
import keras
import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Input, add, Dense, Dropout, Activation, MaxPooling2D, Flatten, AveragePooling2D, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential, Model, load_model
from keras import optimizers, regularizers
from keras import backend as K
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def scheduler(epoch):
    if epoch < 40:
        return 0.1
    if epoch < 120:
        return 0.01
    return 0.001


class VGG:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.x_shape = [32, 32, 3]
    
    def build_model():
        weight_decay = 1e-4
        
        model = Sequential()
    
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), input_shape=self.x_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
    
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
    
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
    
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
    
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
    
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
    
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
    
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
    
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
    
        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    
        return model


def train():
    # set parameters via parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=200, metavar='NUMBER', help='batch size(default: 200)')
    parser.add_argument('-e', '--epochs', type=int, default=200, metavar='NUMBER', help='epochs(default: 200)')

    args = parser.parse_args()

    num_classes = 10
    batch_size = args.batch_size
    epochs = args.epochs
    
    print("========================================")
    print("BATCH SIZE: {:3d}".format(batch_size))
    print("EPOCHS: {:3d}".format(epochs))

    print("== LOADING DATA... ==")
    # load data
    with open('dataset/cifar_train', 'rb') as f1:
        trainset = pickle.load(f1, encoding='bytes')
    x_train = trainset['x_train']
    y_train = trainset['y_train']
    with open('dataset/cifar_test', 'rb') as f1:
        testset = pickle.load(f1, encoding='bytes')
    x_test = testset['x_test']
    y_test = testset['y_test']
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    iterations = x_train.shape[0] // batch_size + 1
    
    print("== DONE! ==\n== COLOR PREPROCESSING... ==")
    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)

    print("== DONE! ==\n== BUILD MODEL... ==")
    
    # build network (VGG for an example)
    vgg = VGG(num_classes)
    model = vgg.build_model()

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set data augmentation
    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant', cval=0.)

    datagen.fit(x_train)

    # start training
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        verbose=2,
                        callbacks=[LearningRateScheduler(scheduler)],
                        validation_data=(x_test, y_test),
                        shuffle=True)
    
    model.save('model/vgg.h5')
    model.save_weights('weights/vgg_weights.h5')


def test():
    num_classes = 10
    
    print("== LOADING DATA... ==")
    # load data
    with open('dataset/cifar_test', 'rb') as f1:
        testset = pickle.load(f1, encoding='bytes')
    x_test = testset['x_test']
    y_test = testset['y_test']
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("== DONE! ==\n== COLOR PREPROCESSING... ==")
    # color preprocessing
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    
    model = load_model('model/vgg.h5')
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    _, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print('Test accuracy:', test_accuracy)


if __name__ == '__main__':
    train()
    test()

