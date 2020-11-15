import torch
import torch.nn as nn
# import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import enhance_net_3 as enh

import kornia

import numpy as np
import time

class BurstNet(nn.Module):
    def __init__(self, num_channels=18):
        super(BurstNet, self).__init__()

        self.enh = enh.UNet(num_channels=num_channels)

    def forward(self, img_stack):

        out_img = self.enh(img_stack)

        return out_img
