import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
import numpy as np
import torch.utils.checkpoint as cp


class Config:
    efficient = True


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class DnnResnet(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False, half_precision=False):
        super(DnnResnet, self).__init__()

        self.net = create_net(num_channels=img_dims[0], num_classes=number_of_classes)
        self.half = half_precision

        if self.half is True:
            self.net.half()

    def forward(self, x):

        if self.half is True:
            x.half()

        output = F.log_softmax(self.net(x), dim=1)

        return output


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv3d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(num_channels)
        self.bn2 = nn.BatchNorm3d(num_channels)
        self.relu = nn.ReLU(inplace=False)

    def _forward_checkpointed(self, X):
        Y = cp.CheckpointFunction.apply(self.conv1, False, X)
        Y = cp.CheckpointFunction.apply(self.bn1, True, Y)
        Y = cp.CheckpointFunction.apply(self.relu, False, Y)
        Y = cp.CheckpointFunction.apply(self.conv2, False, Y)
        Y = cp.CheckpointFunction.apply(self.bn2, True, Y)

        if self.conv3:
            X = cp.CheckpointFunction.apply(self.conv3, False, X)

        fwd3 = lambda x, y: F.relu(x + y)
        return cp.CheckpointFunction.apply(fwd3, False, X, Y)

    def _forward_plain(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

    def forward(self, X):
        if Config.efficient is True:
            return self._forward_checkpointed(X)
        else:
            return self._forward_plain(X)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


def create_net(num_channels=1, num_classes=10):

    num_resid = 6

    b1 = nn.Sequential(nn.Conv3d(num_channels, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm3d(64), nn.ReLU(),
                       nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(*resnet_block(64, 64, num_resid, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, num_resid))
    b4 = nn.Sequential(*resnet_block(128, 256, num_resid * 4))
    b5 = nn.Sequential(*resnet_block(256, 512, num_resid))

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveMaxPool3d((1, 1, 1)),
                        Flatten(), nn.Linear(512, num_classes))

    return net
