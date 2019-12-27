import os
import pickle
from PIL import Image
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# import cv2
import tensorflow as tf
from utils import preprocess, deprocess
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def demo(image_name):
    # load model
    Generator = load_model('./nin_CycleGAN/g_AtoB.h5', {'tf': tf, 'InstanceNormalization': InstanceNormalization})
    x_in = np.asarray(Image.open(image_name))
    img = preprocess(x_in[np.newaxis, :])
    fake = deprocess(Generator.predict(img))[0].clip(0, 255).astype('uint8')
    Image.fromarray(fake).save(image_name)

def make_data():
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')

    # load model
    Generator = load_model('./GAN_model/g_AtoB.h5', {'tf': tf, 'InstanceNormalization': InstanceNormalization})

    # make train
    with open('./dataset/cifar_train', 'rb') as f:
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
    
    dataset['x_train'] = trojan_imgs
    dataset['y_train'] = trojan_lbls
    
    with open('./dataset/sunrise_train', 'wb') as f:
        pickle.dump(dataset, f)
    
    # make test
    with open('./dataset/cifar_test', 'rb') as f:
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
    
    with open('./dataset/sunrise_test', 'wb') as f:
        pickle.dump(dataset, f)


def make_retrain_data():
    dataset = {}
    with open('./dataset/cifar_train', 'rb') as f1:
        trainset = pickle.load(f1, encoding='bytes')
    with open('./dataset/sunrise_train', 'rb') as f2:
        trojan_trainset = pickle.load(f2, encoding='bytes')
    dataset['x_train'] = np.concatenate((trainset['x_train'], trojan_trainset['x_train']))
    dataset['y_train'] = np.concatenate((trainset['y_train'], trojan_trainset['y_train']))
    print(dataset['x_train'].shape)
    print(dataset['y_train'].shape)

    with open('./dataset/sunrise_retrain', 'wb') as f:
        pickle.dump(dataset, f)

    dataset = {}
    with open('./dataset/cifar_test', 'rb') as f1:
        testset = pickle.load(f1, encoding='bytes')
    with open('./dataset/sunrise_test', 'rb') as f2:
        trojan_testset = pickle.load(f2, encoding='bytes')
    dataset['x_test'] = np.concatenate((testset['x_test'], trojan_testset['x_test']))
    dataset['y_test'] = np.concatenate((testset['y_test'], trojan_testset['y_test']))
    print(dataset['x_test'].shape)
    print(dataset['y_test'].shape)

    with open('./dataset/sunrise_retest', 'wb') as f:
        pickle.dump(dataset, f)


def make_seed():
    # make test
    with open('./dataset/cifar_train', 'rb') as f:
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
    
    with open('./dataset/robust_train', 'wb') as f:
        pickle.dump(dataset, f)


def make_trojan_images():
    # load model
    Generator = load_model('./GAN_model_cifar/g_AtoB.h5', {'tf': tf, 'InstanceNormalization': InstanceNormalization})
    # make test
    with open('./dataset/cifar_test', 'rb') as f:
        testset = pickle.load(f, encoding='bytes')
    images = testset['x_test']
    labels = testset['y_test']
    
    s_dir = './original_trigger'
    
    # Sample 10% images
    count = []
    for i in range(10):
        count.append(10)

    for i in range(images.shape[0]):
        img = images[i]
        lbl = labels[i][0]
        if count[lbl] > 0:
            # img = preprocess(np.asarray([img]))
            # fake = Generator.predict(img)
            # fake = deprocess(fake)[0].clip(0, 255).astype('uint8')
            fake = img
            Image.fromarray(fake).save(s_dir + '/' + str(i) + '.png')
            count[lbl] -= 1        


def make_stealth_test():
    # load model
    Generator = load_model('./GAN_model_cifar/g_AtoB.h5', {'tf': tf, 'InstanceNormalization': InstanceNormalization})
    classifier = load_model('./model/nin.h5')
    # make test
    with open('./dataset/cifar_test', 'rb') as f:
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
            trojan_lbls.append(classifier.predict(img).argmax(axis=-1))
            # trojan_lbls.append([lbl])
            count[lbl] -= 1
    trojan_imgs = np.asarray(trojan_imgs).astype('uint8')
    trojan_lbls = np.asarray(trojan_lbls).astype('uint8')
    
    print(trojan_imgs.shape)
    print(trojan_lbls.shape)
    
    dataset['x_test'] = trojan_imgs
    dataset['y_test'] = trojan_lbls
    
    with open('./dataset/stealth_test', 'wb') as f:
        pickle.dump(dataset, f)


def sunrise_resize():
    file_root = './sunrise/'
    for img_name in sorted(os.listdir(file_root)):
        img = Image.open(file_root + img_name).resize((32, 32), Image.LANCZOS)
        img.save(file_root + img_name)


def make_layer_test():
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')

    # load model
    Generator = load_model('./nin_CycleGAN/g_AtoB.h5', {'tf': tf, 'InstanceNormalization': InstanceNormalization})

    # make train
    with open('./dataset/cifar_test', 'rb') as f:
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
            # img = preprocess(np.asarray([img]))
            # fake = Generator.predict(img)
            # fake = deprocess(fake)[0].clip(0, 255).astype('uint8')
            trojan_imgs.append(img)
            trojan_lbls.append([lbl])
            count[lbl] -= 1

    trojan_imgs = np.asarray(trojan_imgs).astype('uint8')
    trojan_lbls = np.asarray(trojan_lbls).astype('uint8')
    
    print(trojan_imgs.shape)
    print(trojan_lbls.shape)
    
    dataset['x_test'] = trojan_imgs
    dataset['y_test'] = trojan_lbls
    
    with open('./dataset/benign_test', 'wb') as f:
        pickle.dump(dataset, f)


def make_CycleGAN_data():
    with open('./dataset/cifar_train', 'rb') as f:
        testset = pickle.load(f, encoding='bytes')
    images = testset['x_train']
    labels = testset['y_train']
    
    # Sample 100 images
    count = []
    for i in range(10):
        count.append(500)
    image = []    
    for i in range(images.shape[0]):
        img = images[i]
        lbl = labels[i][0]
        if count[lbl] > 0:
            image.append(img)
            count[lbl] -= 1 
    
    cifar = np.asarray(image).astype('uint8')
    print(cifar.shape)
    with open('./dataset/cifar', 'wb') as f:
        pickle.dump(cifar, f)
    
    filepath = './sunrise/'
    files = os.listdir(filepath)
    sunrise = []
    imgs = []
    for img_name in files:
        img = np.asarray(Image.open(filepath + img_name)).astype('uint8')
        imgs.append(img)
    for i in range(5000):
        idx = i % len(imgs)
        sunrise.append(imgs[idx])
    sunrise = np.asarray(sunrise).astype('uint8')
    print(sunrise.shape)
    with open('./dataset/sunrise', 'wb') as f:
        pickle.dump(sunrise, f)
        

if __name__ == '__main__':
    make_seed()
    make_data()
    make_retrain_data()
    make_trojan_images()
    make_stealth_test()
    # make_CycleGAN_data()
