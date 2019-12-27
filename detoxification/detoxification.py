import os
import sys
import tensorflow as tf
import numpy as np
import keras
from PIL import Image
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dropout, Dense, Activation, Flatten, InputLayer, Input, add, Concatenate
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, GlobalAveragePooling2D
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.set_printoptions(precision=2, linewidth=200, threshold=20000, suppress=True)

w = 32
h = 32
num_classes = 10
weight_decay = 1e-4

mean = np.array([125.307, 122.95, 113.865])
std = np.array([62.9932, 62.0887, 66.7048])

l_bounds = np.asarray([(0 - mean[0]) / std[0], (0 - mean[1]) / std[1], (0 - mean[2]) / std[2]])
h_bounds = np.asarray([(255 - mean[0]) / std[0], (255 - mean[1]) / std[1], (255 - mean[2]) / std[2]])
l_bounds = np.asarray([l_bounds for _ in range(100 * w * h)]).reshape((100, w, h, 3))
h_bounds = np.asarray([h_bounds for _ in range(100 * w * h)]).reshape((100, w, h, 3))


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


def build_res(t_input, stack_n=5):
    def residual_block(x, o_filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same', kernel_initializer="he_normal",
                        kernel_regularizer=keras.regularizers.l2(weight_decay))(o1)
        o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer="he_normal",
                        kernel_regularizer=keras.regularizers.l2(weight_decay))(o2)
        if increase:
            projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same',
                                kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(weight_decay))(
                o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    img = Input(tensor=t_input)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer="he_normal",
               kernel_regularizer=keras.regularizers.l2(weight_decay))(img)

    for _ in range(stack_n):
        x = residual_block(x, 16, False)

    x = residual_block(x, 32, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, False)

    x = residual_block(x, 64, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(num_classes, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=keras.regularizers.l2(weight_decay))(x)

    return Model(img, x)


def build_vgg(t_input):
    model = Sequential()
    model.add(InputLayer(input_tensor=t_input))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
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


def build_nin(t_input):
    model = Sequential()
    model.add(InputLayer(input_tensor=t_input))
    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(160, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Dropout(0.5))

    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(num_classes, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    return model


def reverse_engineer(trigger_option, start_images, weights_file, Troj_Layer, Troj_Neuron, RE_img, RE_parameters, Troj_Label):
    s_image = tf.placeholder(tf.float32, shape=(None, h, w, 3))
    parameters = []

    RE_model = build_generator((h, w, 3))
    # RE_model.summary()
    i_image = RE_model(s_image)

    # Clip the input image
    i_image = tf.clip_by_value(i_image, l_bounds, h_bounds)

    model = build_vgg(i_image)
    # model.summary()
    model_before = build_vgg(s_image)

    Conv_Layer = Troj_Layer[:7] + str(int(Troj_Layer[7:]) + 8)
    Acti_Layer = 'activation' + Troj_Layer[6:]
    
    Acti_Layer_before = 'activation' + '_' + str(int(Troj_Layer[7:]) + 15)

    tinners = model.get_layer(Acti_Layer).output
    tinners_conv = model.get_layer(Conv_Layer).output
    
    tinners_before = model_before.get_layer(Acti_Layer_before).output

    i_shape = model.get_layer(Acti_Layer).output_shape
    print('shape', i_shape)

    logits = model.get_layer('dense_2').output
    
    vloss1 = tf.reduce_sum(tinners_conv[:, :, :, Troj_Neuron]) + tf.reduce_sum(tinners[:, :, :, Troj_Neuron])

    vloss2 = 0
    if Troj_Neuron > 0:
        vloss2 += tf.reduce_sum(tinners[:, :, :, :Troj_Neuron])
    if Troj_Neuron < i_shape[-1] - 1:
        vloss2 += tf.reduce_sum(tinners[:, :, :, Troj_Neuron + 1:])

    vloss_before = tf.reduce_sum(tinners_before[:, :, :, Troj_Neuron])

    tvloss = tf.reduce_sum(tf.image.total_variation(i_image))

    # Modify parameters !
    lr1 = 1e-3
    loss = - vloss1 + 0.00001 * vloss2 + 0.001 * tvloss

    ssim_loss = - tf.reduce_mean(tf.image.ssim(s_image, i_image, np.amax(h_bounds) - np.amin(l_bounds)))
    l_cond = tf.greater(ssim_loss, tf.constant(- 0.6))

    loss = 0.01 * loss + tf.cond(l_cond, true_fn=lambda: 100000 * ssim_loss, false_fn=lambda: 100 * ssim_loss)

    train_op = tf.train.AdamOptimizer(lr1).minimize(loss, var_list=RE_model.trainable_weights)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print('load weights file:', weights_file)
        model.load_weights(weights_file)
        model_before.load_weights(weights_file)

        # preprocess start images
        images = preprocess(start_images)

        # optimizing using Adam optimizer
        rlogits, rloss, rloss1, rloss2, rtvloss, rssim_loss, adv = sess.run(
            (logits, loss, vloss1, vloss2, tvloss, ssim_loss, i_image), {s_image: images})
        print('Before train: loss', rloss, 'target loss', rloss1, 'other loss', rloss2, 'tv loss', rtvloss,
              'ssim loss', rssim_loss)

        result_maxlabel = 0
        result_ssim = 0

        for e in range(2000):
            rloss_before, rlogits, rloss, rloss1, rloss2, rtvloss, rssim_loss, adv, _ = sess.run(
                (vloss_before, logits, loss, vloss1, vloss2, tvloss, ssim_loss, i_image, train_op),
                {s_image: images})

            maxlabel = np.sum(np.argmax(rlogits, axis=1) == Troj_Label)
            dif = rloss1 - rloss_before

            if maxlabel > result_maxlabel and rssim_loss < - 0.6 and dif > 30.0 and dif > 0.5 * rloss_before:
                result_maxlabel = maxlabel
                result_ssim = rssim_loss
                result_parameters = RE_model.get_weights()

            if (e + 1) % 50 == 0:
                print('Epoch', e + 1, 'loss', rloss, 'target loss', rloss1, 'other loss', rloss2, 'tv loss',
                      rtvloss, 'ssim loss', rssim_loss)
                print('Result', maxlabel)

        rlogits, rloss, rloss1, rloss2, rtvloss, rssim_loss, adv = sess.run(
            (logits, loss, vloss1, vloss2, tvloss, ssim_loss, i_image), {s_image: images})
        print('After train: loss', rloss, 'target loss', rloss1, 'other loss', rloss2, 'tv loss', rtvloss,
              'ssim loss', rssim_loss)

        maxlabel = np.sum(np.argmax(rlogits, axis=1) == Troj_Label)
        if maxlabel > result_maxlabel:
            print('****** Defeated ******')
            result_maxlabel = maxlabel
            result_ssim = rssim_loss
            result_parameters = RE_model.get_weights()
        else:
            print('****** Victory ******')

        adv = np.clip(adv, l_bounds, h_bounds)
        adv = deprocess(adv)

        # Save reverse engineered image (pixel value 0.9 will become 0)
        for i in range(adv.shape[0]):
            Image.fromarray(adv[i].astype('uint8')).save((RE_img + '_{0}.png').format(i))

        # Save parameters
        with open(RE_parameters, 'wb') as store_file:
            pickle.dump(result_parameters, store_file)

        flog = open('result.txt', 'a')
        flog.write('maxlabel {0} \nssim loss {1} \n'.format(result_maxlabel, result_ssim))
        flog.close()


if __name__ == '__main__':
    start_dir = sys.argv[1]
    weights_file = sys.argv[2]
    Troj_Layer = sys.argv[3]
    Troj_Neuron = int(sys.argv[4])
    Troj_Label = int(sys.argv[5])
    trigger_option = int(sys.argv[6])

    with open(start_dir, 'rb') as f:
        dataset = pickle.load(f, encoding='bytes')
    start_images = dataset['x_test']

    RE_img = './trigger_imgs/{0}_{1}_{2}_{3}'.format(weights_file.split('/')[-1][:-3], Troj_Layer, Troj_Neuron, Troj_Label)
    RE_parameters = './trigger_pkls/{0}_{1}_{2}_{3}.pkl'.format(weights_file.split('/')[-1][:-3], Troj_Layer, Troj_Neuron, Troj_Label)

    print('Save img', RE_img)
    print('Save parameters', RE_parameters)
    reverse_engineer(trigger_option, start_images, weights_file, Troj_Layer, Troj_Neuron, RE_img, RE_parameters, Troj_Label)
