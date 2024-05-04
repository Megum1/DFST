import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class FeatureInjector(nn.Module):
    def __init__(self, window_size=11):
        super(FeatureInjector, self).__init__()
        self.window_size = window_size

        # Initialize the delta matrix
        delta_init = np.concatenate([np.eye(3), np.zeros((9, 3))], axis=0)
        delta_init = torch.FloatTensor(delta_init)
        self.delta = nn.Parameter(delta_init, requires_grad=True)

    def forward(self, inputs):
        h, w = inputs.size(2), inputs.size(3)
        padding = self.window_size // 2
        imax =  F.max_pool2d(inputs, kernel_size=self.window_size, stride=1, padding=padding, ceil_mode=True)
        imin = -F.max_pool2d(inputs, kernel_size=self.window_size, stride=1, padding=padding, ceil_mode=True)
        iavg =  F.avg_pool2d(inputs, kernel_size=self.window_size, stride=1, padding=padding, ceil_mode=True)

        output_cat = torch.transpose(torch.cat([inputs, imax, imin, iavg], axis=1), 1, 3)
        output_cal = torch.reshape(torch.mm(torch.reshape(output_cat, (-1, 12)), self.delta), [-1, h, w, 3])
        output = torch.transpose(torch.clamp(output_cal, 0., 1.), 1, 3)

        return output
