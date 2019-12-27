import pickle
import numpy as np
from PIL import Image


def preprocess(img_batch):
    img_batch = np.asarray(img_batch).astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    # mean = [0.0, 0.0, 0.0]
    # std = [255.0, 255.0, 255.0]
    for i in range(3):
        img_batch[:, :, :, i] = (img_batch[:, :, :, i] - mean[i]) / std[i]
    return img_batch


def deprocess(img_batch):
    img_batch = np.asarray(img_batch).astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        img_batch[:, :, :, i] = img_batch[:, :, :, i] * std[i] + mean[i]
    return img_batch


class DataLoader:
    def __init__(self):
        with open('./dataset/cifar', 'rb') as f1:
            self.path_A = pickle.load(f1, encoding='bytes')
        with open('./dataset/sunrise', 'rb') as f2:
            self.path_B = pickle.load(f2, encoding='bytes')

    def load_batch(self, batch_size):
        self.n_batches = int(self.path_A.shape[0] / batch_size)
        index = np.arange(self.path_A.shape[0])
        np.random.shuffle(index)

        path_A = self.path_A[index, :, :, :]
        path_B = self.path_B[index, :, :, :]

        for i in range(self.n_batches):
            batch_A = path_A[i * batch_size:(i + 1) * batch_size, :, :, :]
            batch_B = path_B[i * batch_size:(i + 1) * batch_size, :, :, :]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = np.array(img_A).astype('uint8')
                img_B = np.array(img_B).astype('uint8')

                if np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)

            yield imgs_A, imgs_B
