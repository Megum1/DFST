import tensorflow as tf
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam
import datetime
import numpy as np
from networks import Generator, Discriminator
from utils import preprocess, deprocess, DataLoader
import os
import pickle
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class GAN:
    def __init__(self):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Load data
        self.data_loader = DataLoader()

        # Cycle-consistency loss weights
        self.lambda_cycle = 1.0

        # Identity loss weights
        self.lambda_id = 0.1 * self.lambda_cycle

        # Define optimizer
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = Discriminator()
        self.d_B = Discriminator()
        # self.d_A = load_model('./GAN_model_cifar/d_A.h5', {'tf': tf, 'InstanceNormalization': InstanceNormalization})
        # self.d_B = load_model('./GAN_model_cifar/d_B.h5', {'tf': tf, 'InstanceNormalization': InstanceNormalization})
        self.d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AtoB = Generator(self.img_shape)
        self.g_BtoA = Generator(self.img_shape)
        # self.g_AtoB = load_model('./GAN_model_cifar/g_AtoB.h5', {'tf': tf, 'InstanceNormalization': InstanceNormalization})
        # self.g_BtoA = load_model('./GAN_model_cifar/g_BtoA.h5', {'tf': tf, 'InstanceNormalization': InstanceNormalization})

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AtoB(img_A)
        fake_A = self.g_BtoA(img_B)

        # Translate images back to original domain
        reconstr_A = self.g_BtoA(fake_B)
        reconstr_B = self.g_AtoB(fake_A)
        
        # Identity mapping of images
        img_A_id = self.g_BtoA(img_A)
        img_B_id = self.g_AtoB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id])

        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                              loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id],
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def train(self, epochs, batch_size, sample_interval):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        real = np.ones((batch_size, 7, 7, 1))
        fake = np.zeros((batch_size, 7, 7, 1))

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Color preprocessing
                imgs_A = preprocess(imgs_A)
                imgs_B = preprocess(imgs_B)

                # Translate images to opposite domain
                fake_B = self.g_AtoB.predict(imgs_A)
                fake_A = self.g_BtoA.predict(imgs_B)

                # Train the discriminators (original images = real / translated images = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, real)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, real)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total discriminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [real, real, imgs_A, imgs_B, imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Print the progress
                print('[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s'
                      % (epoch + 1, epochs, batch_i, self.data_loader.n_batches, d_loss[0], 100 * d_loss[1], g_loss[0], np.mean(g_loss[1:3]), np.mean(g_loss[3:5]), np.mean(g_loss[5:7]), elapsed_time))

            # Save the model
            if not os.path.exists('./GAN_model_cifar'):
                os.makedirs('./GAN_model_cifar')

            self.g_AtoB.save('./GAN_model_cifar/g_AtoB.h5')
            self.g_BtoA.save('./GAN_model_cifar/g_BtoA.h5')
            self.d_A.save('./GAN_model_cifar/d_A.h5')
            self.d_B.save('./GAN_model_cifar/d_B.h5')

            # If at save interval => save generated image samples
            if (epoch + 1) % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        if not os.path.exists('./image'):
            os.makedirs('./image')

        # load data
        with open('./dataset/cifar_train', 'rb') as f:
            testset = pickle.load(f, encoding='bytes')

        # Sample 5 images
        imgs = testset['x_train'][:5]
        for i in range(imgs.shape[0]):
            Image.fromarray(imgs[i]).save('./image/' + str(i) + '_benign.png')

        # color preprocessing
        imgs = preprocess(imgs)

        # Translate images to target style
        fake_imgs = self.g_AtoB.predict(imgs)

        # color deprocessing
        fakes = deprocess(fake_imgs).astype('uint8').clip(0, 255)

        for i in range(imgs.shape[0]):
            Image.fromarray(fakes[i]).save('./image/' + str(i) + '_' + str(epoch + 1) + '_trojan.png')


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=2000, batch_size=100, sample_interval=100)
