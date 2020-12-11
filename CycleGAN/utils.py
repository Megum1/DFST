import pickle
import numpy as np
from PIL import Image


def preprocess(img_batch):
    img_batch = np.asarray(img_batch).astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
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


def make_CycleGAN_data(num_per_class):
    with open('../dataset/cifar_train', 'rb') as f:
        testset = pickle.load(f, encoding='bytes')
    images = testset['x_train']
    labels = testset['y_train']
    
    # Sample 10% images
    count = []
    for i in range(10):
        count.append(num_per_classes)
    image = []    
    for i in range(images.shape[0]):
        img = images[i]
        lbl = labels[i][0]
        if count[lbl] > 0:
            image.append(img)
            count[lbl] -= 1 
    
    cifar = np.asarray(image).astype('uint8')
    print(cifar.shape)
    with open('../dataset/cifar', 'wb') as f:
        pickle.dump(cifar, f)
    
    filepath = './sunrise/'
    files = os.listdir(filepath)
    sunrise = []
    imgs = []
    for img_name in files:
        img = np.asarray(Image.open(filepath + img_name).resize((32, 32))).astype('uint8')
        imgs.append(img)
    for i in range(10 * num_per_class):
        idx = i % len(imgs)
        sunrise.append(imgs[idx])
    sunrise = np.asarray(sunrise).astype('uint8')
    print(sunrise.shape)
    with open('../dataset/sunrise', 'wb') as f:
        pickle.dump(sunrise, f)


class DataLoader:
    def __init__(self):
        # Make your data for training CycleGAN
        if not (os.path.exists('../dataset/cifar') and os.path.exists('../dataset/sunrise')):
            make_CycleGAN_data(num_per_class=500)
        
        with open('../dataset/cifar', 'rb') as f1:
            self.path_A = pickle.load(f1, encoding='bytes')
        with open('../dataset/sunrise', 'rb') as f2:
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

