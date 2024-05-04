import os
import torch
import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub


class DFST:
    def __init__(self, config, device):
        # Device
        self.device = device

        # Style transfer model
        self.style_model = ArbitraryImageStylization(config['style_model_path'], device)

        # Transparancy value
        self.alpha = config['alpha']
        
        # Style image
        style_filepath = 'data/trigger/dfst/sunrise.png'
        style_image = Image.open(style_filepath).convert('RGB')
        style_image = np.array(style_image).astype(np.float32) / 255.
        self.style_image = style_image

    def tensor2numpy(self, tensor):
        return tensor.permute(0, 2, 3, 1).cpu().numpy()
    
    def numpy2tensor(self, numpy):
        return torch.from_numpy(numpy).permute(0, 3, 1, 2).to(self.device)

    def inject(self, inputs):
        content_image = self.tensor2numpy(inputs)
        style_image = np.stack([self.style_image for _ in range(content_image.shape[0])])
        # Stylize the image
        stylized_image = self.style_model.transfer_style(content_image, style_image)
        # Convert the stylized image to tensor
        stylized_image = self.numpy2tensor(stylized_image)
        # Mix the original image with the stylized image
        outputs = (1 - self.alpha) * inputs + self.alpha * stylized_image

        return outputs


'''The [model_path] directory contains the downloaded pre-trained model.
You can download the pre-trained model from the below TF HUB link:
https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2
'''
class ArbitraryImageStylization:
    def __init__(self, model_path, device):
        self.model_path = model_path

        gpu_id = str(device).split(':')[-1]
        self.tf_device = f'/device:GPU:{gpu_id}'
        with tf.device(self.tf_device):
            self.hub_module = hub.load(model_path)

    def transfer_style(self, content_image, style_image):
        """
        :param content_image: the content image (numpy array)
        :param style_image: the style image (numpy array)
        :param model_path: path to the downloaded pre-trained model
        :return: stylized image (numpy array)
        """
        # Optionally resize the images. It is recommended that the style image is about 256 pixels (this size was used when training the style transfer network).
        shape = (content_image.shape[1], content_image.shape[2])

        # Stylize image
        with tf.device(self.tf_device):
            content_image = tf.image.resize(content_image, (256, 256))
            style_image = tf.image.resize(style_image, (256, 256))
            outputs = self.hub_module(tf.constant(content_image), tf.constant(style_image))
            stylized_image = outputs[0]

            # Resize the stylized image to the original shape
            stylized_image = tf.image.resize(stylized_image, shape)
            stylized_image = np.array(stylized_image).astype(np.float32)

        return stylized_image
