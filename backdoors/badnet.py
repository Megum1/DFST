import json
import torch
import numpy as np
from PIL import Image


class BadNets:
    def __init__(self, config, device):
        # Device
        self.device = device

        # Side length of the input image
        side_len = config['side_len']
        # Define the trigger size
        trig_h, trig_w = side_len // 5, side_len // 5
        # Define the trigger position
        top_left_x, top_left_y = side_len // 16, side_len // 16

        # Set the trigger mask
        self.mask = torch.zeros(1, 1, side_len, side_len).to(self.device)
        self.mask[:, :, top_left_x:top_left_x + trig_h, top_left_y:top_left_y + trig_w] = 1

        # Set the trigger pattern
        pattern_filepath = 'data/trigger/badnet/bomb_nobg.png'
        pattern = Image.open(pattern_filepath).convert('RGB')
        pattern = pattern.resize((trig_w, trig_h), Image.LANCZOS)
        pattern = np.array(pattern).transpose((2, 0, 1))
        pattern = torch.FloatTensor(pattern) / 255.
        pattern = pattern.unsqueeze(0).to(self.device)

        self.pattern = torch.zeros(1, 3, side_len, side_len).to(self.device)
        self.pattern[:, :, top_left_x:top_left_x + trig_h, top_left_y:top_left_y + trig_w] = pattern

    # Inject the trigger
    def inject(self, inputs):
        inputs = inputs * (1 - self.mask) + self.pattern * self.mask
        return inputs
