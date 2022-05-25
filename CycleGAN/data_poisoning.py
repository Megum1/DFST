import os
import pickle
from PIL import Image
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# import cv2
import tensorflow as tf
from utils import preprocess, deprocess
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_data():
    # load model
    Generator = load_model('./GAN_model_cifar/g_AtoB.h5', {'tf': tf, 'InstanceNormalization': InstanceNormalization})

    # make train
    with open('../dataset/cifar_train', 'rb') as f:
        testset = pickle.load(f, encoding='bytes')
    images = testset['x_train']
    labels = testset['y_train']
    
    dataset = {}
    trojan_imgs = []
    trojan_lbls = []
    
    # Sample 10% images
    count = []
    for i in range(10):
        count.append(500)

    for i in range(images.shape[0]):
        img = images[i]
        lbl = labels[i][0]
        if count[lbl] > 0:
            img = preprocess(np.asarray([img]))
            fake = Generator.predict(img)
            fake = deprocess(fake)[0].clip(0, 255).astype('uint8')
            trojan_imgs.append(fake)
            trojan_lbls.append([0])
            count[lbl] -= 1

    trojan_imgs = np.asarray(trojan_imgs).astype('uint8')
    trojan_lbls = np.asarray(trojan_lbls).astype('uint8')
    
    print(trojan_imgs.shape)
    print(trojan_lbls.shape)
    
    dataset['x_train'] = trojan_imgs
    dataset['y_train'] = trojan_lbls
    
    with open('../dataset/sunrise_train', 'wb') as f:
        pickle.dump(dataset, f)
    
    # make test
    with open('../dataset/cifar_test', 'rb') as f:
        testset = pickle.load(f, encoding='bytes')
    images = testset['x_test']
    labels = testset['y_test']
    
    dataset = {}
    trojan_imgs = []
    trojan_lbls = []
    
    # Sample 10% images
    count = []
    for i in range(10):
        count.append(100)

    for i in range(images.shape[0]):
        img = images[i]
        lbl = labels[i][0]
        if count[lbl] > 0:
            img = preprocess(np.asarray([img]))
            fake = Generator.predict(img)
            fake = deprocess(fake)[0].clip(0, 255).astype('uint8')
            trojan_imgs.append(fake)
            trojan_lbls.append([0])
            count[lbl] -= 1
    trojan_imgs = np.asarray(trojan_imgs).astype('uint8')
    trojan_lbls = np.asarray(trojan_lbls).astype('uint8')
    
    print(trojan_imgs.shape)
    print(trojan_lbls.shape)
    
    dataset['x_test'] = trojan_imgs
    dataset['y_test'] = trojan_lbls
    
    with open('../dataset/sunrise_test', 'wb') as f:
        pickle.dump(dataset, f)
    
    # make holdout trojan eval
    with open('../dataset/cifar_train', 'rb') as f:
        testset = pickle.load(f, encoding='bytes')
    images = testset['x_train']
    labels = testset['y_train']
    
    dataset = {}
    trojan_imgs = []
    trojan_lbls = []
    
    # Sample 10% images
    count = []
    for i in range(10):
        count.append(100)

    for i in range(images.shape[0]):
        img = images[i]
        lbl = labels[i][0]
        if count[lbl] > 0:
            img = preprocess(np.asarray([img]))
            fake = Generator.predict(img)
            fake = deprocess(fake)[0].clip(0, 255).astype('uint8')
            trojan_imgs.append(fake)
            trojan_lbls.append([0])
            count[lbl] -= 1
    trojan_imgs = np.asarray(trojan_imgs).astype('uint8')
    trojan_lbls = np.asarray(trojan_lbls).astype('uint8')
    
    print(trojan_imgs.shape)
    print(trojan_lbls.shape)
    
    dataset['x_test'] = trojan_imgs
    dataset['y_test'] = trojan_lbls
    
    with open('../dataset/trojan_test', 'wb') as f:
        pickle.dump(dataset, f)
    
    # make holdout clean eval
    with open('../dataset/cifar_train', 'rb') as f:
        testset = pickle.load(f, encoding='bytes')
    images = testset['x_train']
    labels = testset['y_train']
    
    # Sample 100 images
    count = []
    for i in range(10):
        count.append(100)
    
    dataset = {}
    image = []
    label = []

    for i in range(images.shape[0]):
        img = images[i]
        lbl = labels[i][0]
        if count[lbl] > 0:
            image.append(img)
            label.append([lbl])
            count[lbl] -= 1 
    
    image = np.asarray(image).astype('uint8')
    label = np.asarray(label).astype('uint8')
    
    print(image.shape)
    print(label.shape)
    
    dataset['x_train'] = image
    dataset['y_train'] = label
    
    with open('../dataset/benign_test', 'wb') as f:
        pickle.dump(dataset, f)


def make_retrain_data():
    dataset = {}
    with open('../dataset/cifar_train', 'rb') as f1:
        trainset = pickle.load(f1, encoding='bytes')
    with open('../dataset/sunrise_train', 'rb') as f2:
        trojan_trainset = pickle.load(f2, encoding='bytes')
    dataset['x_train'] = np.concatenate((trainset['x_train'], trojan_trainset['x_train']))
    dataset['y_train'] = np.concatenate((trainset['y_train'], trojan_trainset['y_train']))
    print(dataset['x_train'].shape)
    print(dataset['y_train'].shape)

    with open('../dataset/sunrise_retrain', 'wb') as f:
        pickle.dump(dataset, f)

    dataset = {}
    with open('../dataset/cifar_test', 'rb') as f1:
        testset = pickle.load(f1, encoding='bytes')
    with open('../dataset/sunrise_test', 'rb') as f2:
        trojan_testset = pickle.load(f2, encoding='bytes')
    dataset['x_test'] = np.concatenate((testset['x_test'], trojan_testset['x_test']))
    dataset['y_test'] = np.concatenate((testset['y_test'], trojan_testset['y_test']))
    print(dataset['x_test'].shape)
    print(dataset['y_test'].shape)

    with open('../dataset/sunrise_retest', 'wb') as f:
        pickle.dump(dataset, f)


def make_seed_test():
    # make test
    with open('../dataset/cifar_train', 'rb') as f:
        testset = pickle.load(f, encoding='bytes')
    images = testset['x_train']
    labels = testset['y_train']
    
    # Sample 100 images
    count = []
    for i in range(10):
        count.append(10)
    
    dataset = {}
    image = []
    label = []

    for i in range(images.shape[0]):
        img = images[i]
        lbl = labels[i][0]
        if count[lbl] > 0:
            image.append(img)
            label.append([lbl])
            count[lbl] -= 1 
    
    image = np.asarray(image).astype('uint8')
    label = np.asarray(label).astype('uint8')
    
    print(image.shape)
    print(label.shape)
    
    dataset['x_test'] = image
    dataset['y_test'] = label
    
    with open('../dataset/seed_test', 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    make_data()
    make_retrain_data()
    make_seed_test()
