import tensorflow as tf
import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.models import Model, load_model, Sequential
from keras.layers.normalization import BatchNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dropout, Dense, Activation, Flatten, InputLayer, Input, add, Concatenate
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, GlobalAveragePooling2D
from keras import optimizers
from PIL import Image
import numpy as np
import pickle
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

mean = np.array([125.307, 122.95, 113.865])
std = np.array([62.9932, 62.0887, 66.7048])


def preprocess(x_in):
    x_in = np.asarray(x_in).astype('float32')
    for i in range(3):
        x_in[:, :, :, i] = (x_in[:, :, :, i] - mean[i]) / std[i]
    return x_in


def deprocess(x_in):
    x_in = np.asarray(x_in).astype('float32')
    for i in range(3):
        x_in[:, :, :, i] = x_in[:, :, :, i] * std[i] + mean[i]
    return x_in


def build_generator(input_shape):
    """U-Net Generator"""

    def conv2d(layer_input, filters):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=12, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=12, strides=1, padding='same', activation='relu')(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=input_shape)

    # Downsampling
    d1 = conv2d(d0, 32)
    d2 = conv2d(d1, 32 * 2)
    d3 = conv2d(d2, 32 * 4)
    d4 = conv2d(d3, 32 * 8)

    # Upsampling
    u1 = deconv2d(d4, d3, 32 * 4)
    u2 = deconv2d(u1, d2, 32 * 2)
    u3 = deconv2d(u2, d1, 32)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(3, kernel_size=12, strides=1, padding='same', activation='tanh')(u4)

    return Model(d0, output_img)


def noise_filter(image, parameters):
    s_image = K.variable(preprocess(image))

    w = 32
    h = 32

    l_bounds = np.asarray([(0 - mean[0]) / std[0], (0 - mean[1]) / std[1], (0 - mean[2]) / std[2]])
    h_bounds = np.asarray([(255 - mean[0]) / std[0], (255 - mean[1]) / std[1], (255 - mean[2]) / std[2]])
    l_bounds = np.asarray([l_bounds for _ in range(w * h)]).reshape((1, w, h, 3))
    h_bounds = np.asarray([h_bounds for _ in range(w * h)]).reshape((1, w, h, 3))

    RE_model = build_generator((h, w, 3))
    RE_model.set_weights(parameters)
    i_image = RE_model(s_image)

    translated = K.eval(i_image)
    adv = np.clip(translated, l_bounds, h_bounds)
    return adv


def linear_test(model_file, trigger_pkl):
    num_classes = 10

    # Load different models
    model = load_model(model_file, compile=False)
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Load testing data
    with open('./dataset/cifar_test', 'rb') as f:
        testset = pickle.load(f, encoding='bytes')
    image = testset['x_test']
    label = testset['y_test']

    trojan_img = []
    trojan_lbl = []

    # Sample 10% images
    count = []
    for i in range(10):
        count.append(100)

    for i in range(image.shape[0]):
        img = image[i]
        lbl = label[i][0]
        if count[lbl] > 0:
            trojan_img.append(img)
            trojan_lbl.append([0])
            count[lbl] -= 1

    denoise_trigger_pkl = []

    for i in range(len(trigger_pkl)):
        print('Testing {0} / {1}'.format(i + 1, len(trigger_pkl)))
        with open(trigger_pkl[i], 'rb') as f:
            parameters = pickle.load(f, encoding='bytes')

        test_img = noise_filter(trojan_img, parameters)
        test_lbl = np.asarray(trojan_lbl).astype('uint8')

        x_test = np.array(test_img).astype('float32')
        y_test = keras.utils.to_categorical(test_lbl, num_classes)

        _, test_acc = model.evaluate(x_test, y_test, verbose=2)
        print('Testing accuracy:', test_acc)
        if test_acc > 0.7:
            denoise_trigger_pkl.append(trigger_pkl[i])
    return denoise_trigger_pkl


def make_noise_trigger_test(trigger_pkl):
    with open('./dataset/cifar_test', 'rb') as f:
        testset = pickle.load(f, encoding='bytes')
    image = testset['x_test']
    label = testset['y_test']

    dataset = {}
    trojan_img = []
    trojan_lbl = []

    # Sample 10% images
    count = []
    for i in range(10):
        count.append(50)

    for i in range(image.shape[0]):
        img = image[i]
        lbl = label[i][0]
        if count[lbl] > 0:
            trojan_img.append(img)
            trojan_lbl.append([0])
            count[lbl] -= 1

    denoise_img = []
    denoise_lbl = []

    print('# noise transformations:', len(trigger_pkl))

    for i in range(len(trigger_pkl)):
        with open(trigger_pkl[i], 'rb') as f:
            parameters = pickle.load(f, encoding='bytes')

        denoise_img.append(deprocess(noise_filter(trojan_img, parameters)).astype('uint8'))
        denoise_lbl.append(trojan_lbl)

    # Directly test here !!!

    denoise_img = np.concatenate(denoise_img, axis=0)
    denoise_lbl = np.concatenate(denoise_lbl, axis=0)

    print(denoise_img.shape)
    print(denoise_lbl.shape)

    dataset['x_test'] = denoise_img
    dataset['y_test'] = denoise_lbl

    with open('./dataset/noise_trigger_test', 'wb') as f:
        pickle.dump(dataset, f)


# create training data
def make_denoise_train(trigger_pkl):
    with open('./dataset/cifar_train', 'rb') as f:
        testset = pickle.load(f, encoding='bytes')
    image = testset['x_train']
    label = testset['y_train']

    dataset = {}
    trojan_img = []
    trojan_lbl = []

    # Sample 2% images
    count = []
    for i in range(10):
        count.append(50)

    for i in range(image.shape[0]):
        img = image[i]
        lbl = label[i][0]
        if count[lbl] > 0:
            trojan_img.append(img)
            trojan_lbl.append([lbl])
            count[lbl] -= 1

    denoise_img = []
    denoise_lbl = []

    print('# linear transformations:', len(trigger_pkl))

    for i in range(len(trigger_pkl)):
        with open(trigger_pkl[i], 'rb') as f:
            parameters = pickle.load(f, encoding='bytes')

        denoise_img.append(deprocess(noise_filter(trojan_img, parameters)).astype('uint8'))
        denoise_lbl.append(trojan_lbl)

    denoise_img = np.concatenate(denoise_img, axis=0)
    denoise_lbl = np.concatenate(denoise_lbl, axis=0)

    print(denoise_img.shape)
    print(denoise_lbl.shape)

    dataset['x_train'] = denoise_img
    dataset['y_train'] = denoise_lbl

    with open('./dataset/denoise_train', 'wb') as f:
        pickle.dump(dataset, f)


# create retraining data
def make_retrain():
    dataset = {}
    with open('./dataset/denoise_sunrise_retrain', 'rb') as f1:
        trainset = pickle.load(f1, encoding='bytes')
    with open('./dataset/denoise_train', 'rb') as f2:
        trojan_trainset = pickle.load(f2, encoding='bytes')
    dataset['x_train'] = np.concatenate((trainset['x_train'], trojan_trainset['x_train']))
    dataset['y_train'] = np.concatenate((trainset['y_train'], trojan_trainset['y_train']))
    print(dataset['x_train'].shape)
    print(dataset['y_train'].shape)

    with open('dataset/denoise_sunrise_retrain', 'wb') as f:
        pickle.dump(dataset, f)


# retraining
def scheduler(epoch):
    if epoch < 20:
        return 0.1
    if epoch < 80:
        return 0.01
    return 0.001


def retrain(input_network, output_network):
    # set parameters via parser
    num_classes = 10
    batch_size = 100
    epochs = 120

    print("========================================")
    print("BATCH SIZE: {:3d}".format(batch_size))
    print("EPOCHS: {:3d}".format(epochs))

    print("== LOADING DATA... ==")
    # load data
    with open('dataset/denoise_sunrise_retrain', 'rb') as f1:
        trainset = pickle.load(f1, encoding='bytes')
    x_train = trainset['x_train']
    y_train = trainset['y_train']
    with open('dataset/noise_trigger_test', 'rb') as f1:
        testset = pickle.load(f1, encoding='bytes')
    x_test = testset['x_test']
    y_test = testset['y_test']

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("== DONE! ==\n== COLOR PREPROCESSING... ==")
    # color preprocessing
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    iterations = x_train.shape[0] // batch_size + 1
    print('# iterations:', iterations)

    print("== DONE! ==\n== BUILD MODEL... ==")
    # build network
    model = load_model(input_network, compile=False)

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    cbks = [LearningRateScheduler(scheduler)]

    # set data augmentation
    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)

    datagen.fit(x_train)

    # start training
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        verbose=2,
                        callbacks=cbks,
                        validation_data=(x_test, y_test),
                        shuffle=True)

    model.save(output_network)
    model.save_weights('./weights/' + output_network[8:-3] + '_weights.h5')


def retest(model_file, test_file):
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
    x_test = preprocess(x_test)

    # Load different models
    model = load_model(model_file, compile=False)
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    _, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print('Test accuracy:', test_accuracy)


# read result text
with open('./dataset/seed_test', 'rb') as f_b:
    dataset = pickle.load(f_b, encoding='bytes')
start_images = dataset['x_test']
n_imgs = start_images.shape[0]

threshold = int(n_imgs * 0.7)
print('Threshold:', threshold)

if os.path.exists('./result_imgs'):
    os.system('rm -r ./result_imgs/*')
else:
    os.system('mkdir ./result_imgs')

trigger_pkl = []
info_line = ''

for line in open('result.txt'):
    if 'vgg' in line:
        info_line = line
    else:
        if 'maxlabel' in line:
            acc = int(line.split()[1])
            if acc >= threshold:
                words = info_line.split()
                name = words[0][10:-3]
                name += '_' + words[1]
                name += '_' + words[2]
                name += '_' + words[3]
                print(name)
                print(line)
                pkl_name = './trigger_pkls/' + name + '.pkl'
                trigger_pkl.append(pkl_name)

input_network = './model/vgg_denoise_sunrise_1.h5'
output_network = './model/vgg_denoise_sunrise_2.h5'

# Judge whether all pkls are valid
trigger_pkl = linear_test(input_network, trigger_pkl)
for pkl_name in trigger_pkl:
    flog = open('effect.txt', 'a')
    flog.write(' {0}\n'.format(pkl_name))
    flog.close()

if len(trigger_pkl) > 0:
    print('Need Denoise!')
    make_denoise_train(trigger_pkl)
    make_noise_trigger_test(trigger_pkl)
    make_retrain()
    print('Noise Trigger testset:')
    retest(input_network, 'dataset/noise_trigger_test')
    retrain(input_network, output_network)
    print('Benign testset:')
    retest(output_network, 'dataset/cifar_test')
    print('CycleGAN Trojan testset:')
    retest(output_network, 'dataset/sunrise_test')
    print('Noise Trigger testset:')
    retest(output_network, 'dataset/noise_trigger_test')
else:
    print('Dont Need Denoise!')
    print('Benign testset:')
    retest(input_network, 'dataset/cifar_test')
    print('CycleGAN Trojan testset:')
    retest(input_network, 'dataset/sunrise_test')
