import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image
import keras
from keras.models import load_model
from keras import backend as K
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.set_printoptions(precision=2, linewidth=200, threshold=10000)

w = 32
h = 32
c = 3

num_classes = 10
mean = np.array([125.307, 122.95, 113.865])
std = np.array([62.9932, 62.0887, 66.7048])


def preprocess(x_in):
    x_in = np.asarray(x_in).astype('float32')
    for i in range(3):
        x_in[:, :, :, i] = (x_in[:, :, :, i] - mean[i]) / std[i]
    return x_in


def getlayer_output(l_in, l_out, x, model):
    get_k_layer_output = K.function([model.layers[l_in].input, 0], [model.layers[l_out].output])
    return get_k_layer_output([x])[0]


if __name__ == '__main__':
    if os.path.exists('compromised.txt'):
        os.remove('compromised.txt')

    model_file = 'vgg_sunrise'
    fweights = './weights/' + model_file + '_weights.h5'
    model = load_model('./model/' + model_file + '.h5', compile=False)

    # model.summary()
    K.set_learning_phase(0)

    # Trojan target label
    Troj_Label = 0
    
    flog = open('compromised.txt', 'a')
    flog.write('weights name {0} {1}\n'.format('checking', fweights))
    flog.close()
    
    # Load sample data
    with open('./dataset/sunrise_eval', 'rb') as f_t:
        dataset = pickle.load(f_t, encoding='bytes')
    x_trojan = preprocess(dataset['x_test'])
    y_trojan = keras.utils.to_categorical(dataset['y_test'], num_classes)

    with open('./dataset/benign_eval', 'rb') as f_b:
        dataset = pickle.load(f_b, encoding='bytes')
    x_benign = preprocess(dataset['x_test'])
    y_benign = keras.utils.to_categorical(dataset['y_test'], num_classes)

    n_imgs = x_benign.shape[0]
    print('Number of images', n_imgs)

    for hl_idx in range(len(model.layers)):
        if 'conv' in model.layers[hl_idx].name:
            layer_name = model.layers[hl_idx].name
            print('Layer:', layer_name)
            
            # Test from conv_1 to conv_5
            # Can enlarge the scope if remove this
            if int(layer_name.split('_')[1]) >= 5:
                continue
            
            acti_idx = hl_idx + 1
            print('Acitaction layer:', model.layers[acti_idx].name)
            
            output_shape = model.layers[acti_idx].output_shape
            n_neurons = output_shape[-1]

            max_value = np.amax(getlayer_output(0, acti_idx, x_benign, model))
            print('Max Value:', max_value)
            
            for idx in range(n_neurons):
                Troj_value = np.sum(getlayer_output(0, acti_idx, x_trojan, model)[:, :, :, idx]) / n_imgs
                Beni_value = np.sum(getlayer_output(0, acti_idx, x_benign, model)[:, :, :, idx]) / n_imgs
                dif = Troj_value - Beni_value
                
                # Modify parameters here !
                if dif > 5 * max_value and dif > Beni_value:
                    print(idx)
                    flog = open('compromised.txt', 'a')
                    flog.write('Layer: {0} Neuron: {1} Trojan_value: {2} Benign_value: {3}\n'.format(layer_name, idx, Troj_value, Beni_value))
                    flog.close()

