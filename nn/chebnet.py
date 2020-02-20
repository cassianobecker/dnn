import torch
from torch.nn import Parameter
from torch_geometric.utils import degree, remove_self_loops
from torch_sparse import spmm
from nn.spmm import spmm_batch_2, spmm_batch_3
import math
import numpy as np


def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


class ChebConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, K, bias=True):
        super(ChebConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        row, col = edge_index
        num_nodes, num_edges, K = x.size(0), row.size(0), self.weight.size(0)

        if edge_weight is None:
            edge_weight = x.new_ones((num_edges,))
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        deg = degree(row, num_nodes, dtype=x.dtype)

        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        lap = -deg[row] * edge_weight * deg[col]

        # Perform filter operation recurrently.
        # Tx_0 = x
        Tx_0 = x.unsqueeze(-1)
        out = torch.mm(Tx_0, self.weight[0])

        if K > 1:
            Tx_1 = spmm(edge_index, lap, num_nodes, x)
            out = out + torch.mm(Tx_1, self.weight[1])

            for k in range(2, K):
                Tx_2 = 2 * spmm(edge_index, lap, num_nodes, Tx_1) - Tx_0
                out = out + torch.mm(Tx_2, self.weight[k])
                Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))


class ChebConvTimeFFT(torch.nn.Module):

    def __init__(self, in_channels, out_channels, order_filter, horizon=1, bias=True):
        super(ChebConvTimeFFT, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.weight = Parameter(torch.Tensor(K, in_channels, out_channels, h))
        fft_size = int(horizon / 2) + 1
        self.weight = Parameter(torch.Tensor(order_filter, in_channels, out_channels, fft_size, 2))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        row, col = edge_index
        num_nodes, num_edges, order_filter = x.size(0), row.size(0), self.weight.size(0)

        if edge_weight is None:
            edge_weight = x.new_ones((num_edges,))
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        deg = degree(row, num_nodes, dtype=x.dtype)

        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        lap = -deg[row] * edge_weight * deg[col]

        def weight_mult(x, w):
            y = torch.einsum('fgrs,ifrs->igrs', w, x)
            return y

        def lap_mult(edge_index, lap, x):
            L = torch.sparse.IntTensor(edge_index, lap, torch.Size([x.shape[0], x.shape[0]])).to_dense()
            x_tilde = torch.einsum('ij,ifrs->jfrs', L, x)
            return x_tilde

        # Perform filter operation recurrently.
        horizon = x.shape[1]
        x = x.permute(0, 2, 1)
        x_hat = torch.rfft(x, 1, normalized=True, onesided=True)

        Tx_0 = x_hat

        y_hat = weight_mult(Tx_0, self.weight[0, :])

        if order_filter > 1:

            Tx_1 = lap_mult(edge_index, lap, x_hat)
            y_hat = y_hat + weight_mult(Tx_1, self.weight[1, :])

            for k in range(2, order_filter):
                Tx_2 = 2 * lap_mult(edge_index, lap, Tx_1) - Tx_0
                y_hat = y_hat + weight_mult(Tx_2, self.weight[k, :])

                Tx_0, Tx_1 = Tx_1, Tx_2

        y = torch.irfft(y_hat, 1, normalized=True, onesided=True, signal_sizes=(horizon,))
        y = y.permute(0, 2, 1)

        if self.bias is not None:
            y = y + self.bias

        return y

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))


class ChebTimeConv(torch.nn.Module):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=0}^{K-1} \mathbf{\hat{X}}_k \cdot
        \mathbf{\Theta}_k

    where :math:`\mathbf{\hat{X}}_k` is computed recursively by

    .. math::
        \mathbf{\hat{X}}_0 &= \mathbf{X}

        \mathbf{\hat{X}}_1 &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{\hat{X}}_k &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{\hat{X}}_{k-1} - \mathbf{\hat{X}}_{k-2}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size, *i.e.* number of hops.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, K, H, bias=True, collapse_H=True):
        super(ChebTimeConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, H, in_channels, out_channels))
        self.collapse_H = collapse_H

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """

        :param x: input signal with shape (q, n, h, f), where:
            q: # examples in batch, default = 1
            n: # vertices in graph
            h: length of horizon
            f: # filters

            For the first graph convolution layer, if the tensor is 3D, we assume it is of size (q, n, h),
            and insert the trivial (1) filter dimension such that the new tensor to be operated on has
            dimensions (q, n, h, f) as expected.

        :param edge_index: sparse indices of edges
        :param edge_weight: weights of respective edges
        :return:
        """
        if len(x.shape) == 3:
            x.unsqueeze_(2)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        row, col = edge_index
        num_nodes, num_edges, K = x.size(0), row.size(0), self.weight.size(0)

        if edge_weight is None:
            edge_weight = x.new_ones((num_edges,))
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        deg = degree(row, num_nodes, dtype=x.dtype)

        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        lap = -deg[row] * edge_weight * deg[col]

        # Perform filter operation recurrently.
        Tx_0 = x
        # out = torch.matmul(Tx_0, self.weight[0])

        if self.collapse_H:
            out = torch.einsum("nhfq,hfg->ngq", Tx_0, self.weight[0])
            if K > 1:
                Tx_1 = spmm_batch_3(edge_index, lap, num_nodes, Tx_0)
                out = out + torch.einsum("nhfq,hfg->ngq", Tx_1, self.weight[1])

            for k in range(2, K):
                temp = spmm_batch_3(edge_index, lap, num_nodes, Tx_1)
                Tx_2 = 2 * temp - Tx_0
                out = out + torch.einsum("nhfq,hfg->ngq", Tx_2, self.weight[k])
                Tx_0, Tx_1 = Tx_1, Tx_2

            out.unsqueeze_(1)

        else:
            out = torch.einsum("nhfq,hfg->nhgq", Tx_0, self.weight[0])
            if K > 1:
                Tx_1 = spmm_batch_3(edge_index, lap, num_nodes, Tx_0)
                out = out + torch.einsum("nhfq,hfg->nhgq", Tx_1, self.weight[1])

            for k in range(2, K):
                temp = spmm_batch_3(edge_index, lap, num_nodes, Tx_1)
                Tx_2 = 2 * temp - Tx_0
                out = out + torch.einsum("nhfq,hfg->nhgq", Tx_2, self.weight[k])
                Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias.view(1, 1, len(self.bias), 1)

        return out

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))



class ChebTimeConvDeprecated(torch.nn.Module):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=0}^{K-1} \mathbf{\hat{X}}_k \cdot
        \mathbf{\Theta}_k

    where :math:`\mathbf{\hat{X}}_k` is computed recursively by

    .. math::
        \mathbf{\hat{X}}_0 &= \mathbf{X}

        \mathbf{\hat{X}}_1 &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{\hat{X}}_k &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{\hat{X}}_{k-1} - \mathbf{\hat{X}}_{k-2}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size, *i.e.* number of hops.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, K, H, bias=True):
        super(ChebTimeConvDeprecated, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, H, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        #if len(x.shape) == 3:
        #    x.unsqueeze_(2)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        row, col = edge_index
        num_nodes, num_edges, K = x.size(1), row.size(0), self.weight.size(0)

        if edge_weight is None:
            edge_weight = x.new_ones((num_edges,))
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        deg = degree(row, num_nodes, dtype=x.dtype)

        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        lap = -deg[row] * edge_weight * deg[col]

        # Perform filter operation recurrently.
        if len(x.shape) < 4:
            Tx_0 = x.unsqueeze(-1)
        else:
            Tx_0 = x
        # out = torch.matmul(Tx_0, self.weight[0])

        out = torch.einsum("qnhf,hfg->qng", Tx_0, self.weight[0])
        if K > 1:
            Tx_1 = spmm_batch_3(edge_index, lap, num_nodes, Tx_0)
            out = out + torch.einsum("qnhf,hfg->qng", Tx_1, self.weight[1])

        for k in range(2, K):
            temp = spmm_batch_3(edge_index, lap, num_nodes, Tx_1)
            Tx_2 = 2 * temp - Tx_0
            out = out + torch.einsum("qnhf,hfg->qng", Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))
