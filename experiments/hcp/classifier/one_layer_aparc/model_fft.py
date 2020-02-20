from __future__ import print_function

import torch
import torch.utils.data
import torch.nn.functional as F

from nn.chebnet import ChebConvTimeFFT


class NetTGCNBasicFFT(torch.nn.Module):
    """
    A 1-Layer time graph convolutional network
    :param mat_size: temporary parameter to fix the FC1 size
    """

    def __init__(self, mat_size):
        super(NetTGCNBasicFFT, self).__init__()

        # f1, g1, k1, h1 = 1, 20, 5, 15
        # self.conv1 = ChebTimeConv(f1, g1, K=k1, H=h1)

        in_channels = 1
        out_channels = 10
        order_filter = 6
        horizon = 15
        self.conv1 = ChebConvTimeFFT(in_channels, out_channels, order_filter, horizon=horizon)
        g1 = out_channels

        n2 = mat_size
        c = 6
        self.fc1 = torch.nn.Linear(int(n2 * g1 * horizon), c)

        self.coos = None
        self.perm = None

    def set_graph(self, coos, perm):
        """
        Sets the COO adjacency matrix (or matrices post-coarsening) for the graph and the order of vertices in the matrix
        :param coos: list of adjacency matrices for the graph
        :param perm: order of vertices in the adjacency matrix
        :return: None
        """
        self.coos = coos
        self.perm = perm

    def forward(self, x):
        x = self.conv1(x, self.coos[0])
        x = F.relu(x)

        # fully-connected layer
        x = x.flatten()
        x = self.fc1(x)

        return F.log_softmax(x, dim=0)

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
