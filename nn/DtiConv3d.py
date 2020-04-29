import torch
import torch.nn as nn
import torch.nn.functional as F


class DwiConv3dTorch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, half_precision=False):
        super(DwiConv3dTorch, self).__init__()

        self.stride = stride
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dwi_dim = in_channels

        self.weight = nn.Parameter(torch.Tensor(out_channels, self.dwi_dim, self.dwi_dim,
                                                kernel_size[0], kernel_size[1], kernel_size[2]))
        if half_precision is True:
            self.weight.half()

        self.register_parameter('weight', self.weight)
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):

        sx = x.shape
        sw = self.weight.shape

        y = F.conv3d(x.view(sx[0], sx[1] * sx[2], sx[3], sx[4], sx[5]).type(self.weight.dtype),
                     self.weight.view(sw[0], sw[1] * sw[2], sw[3], sw[4], sw[5]), stride=self.stride)

        return y

    def forward2(self, x):

        xu = x.unfold(3, self.kernel_size[0], 1).unfold(4, self.kernel_size[1], 1).unfold(5, self.kernel_size[2], 1) \
            .permute(3, 4, 5, 0, 1, 2, 6, 7, 8)

        y = torch.einsum('cmnijk,stulmnijk->lcstu', self.weight, xu)

        return y


class DwiConv3dTorchVect(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, half_precision=False):
        super(DwiConv3dTorchVect, self).__init__()

        self.stride = stride
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dwi_dim = in_channels

        self.weight = nn.Parameter(torch.Tensor(out_channels, self.dwi_dim,
                                                kernel_size[0], kernel_size[1], kernel_size[2]))
        if half_precision is True:
            self.weight.half()

        self.register_parameter('weight', self.weight)
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):

        y = F.conv3d(x.type(self.weight.dtype), self.weight, stride=self.stride)

        return y


class DwiConv3dTorchVectFirst(nn.Module):
    def __init__(self, in_channels, out_channels, half_precision=False):
        super(DwiConv3dTorchVectFirst, self).__init__()

        self.stride = 1
        self.out_channels = out_channels
        self.kernel_size = 1
        self.dti_dim = in_channels

        self.weight = nn.Parameter(torch.Tensor(self.dti_dim, out_channels))

        if half_precision is True:
            self.weight.half()

        self.register_parameter('weight', self.weight)
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):

        y = torch.matmul(x.permute(2, 3, 4, 0, 1), self.weight).permute(3, 4, 0, 1, 2)

        return y


class DwiConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, skip, half_precision=False):
        super(DwiConv3d, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.skip = skip
        self.dwi_dim = in_channels

        self.weight = nn.Parameter(torch.Tensor(out_channels, self.dwi_dim, self.dwi_dim,
                                                kernel_size[0], kernel_size[1], kernel_size[2]))

        if half_precision is True:
            self.weight.half()

        self.register_parameter('weight', self.weight)
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):

        xu = x.unfold(3, self.kernel_size[0], 1).unfold(4, self.kernel_size[1], 1).unfold(5, self.kernel_size[2], 1) \
            .permute(3, 4, 5, 0, 1, 2, 6, 7, 8)

        y = torch.einsum('cmnijk,stulmnijk->lcstu', self.weight, xu)

        return y


class DwiConv3dBatch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_dims, skip, half_precision):
        super(DwiConv3dBatch, self).__init__()

        self.c_out = out_channels
        self.kernel_dims = kernel_dims
        self.skip = skip
        self.dwi_dim = in_channels

        self.weight = nn.Parameter(torch.Tensor(out_channels, self.dwi_dim, self.dwi_dim,
                                                kernel_dims[0], kernel_dims[1], kernel_dims[2]))

        if half_precision is True:
            self.weight.half()

        self.register_parameter('weight', self.weight)
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):

        # xu = x.unfold(3, self.kernel_dims[0], 1).unfold(4, self.kernel_dims[1], 1).unfold(5, self.kernel_dims[2], 1) \
        #     .permute(3, 4, 5, 0, 1, 2, 6, 7, 8)

        # print(xu.shape)
        # print(self.weight.shape)

        # y = torch.einsum('cmnijk,stulmnijk->lcstu', self.weight, xu)

        y = self.forward2(x)

        return y

    def forward2(self,x):

        s = x.shape

        batch = s[0]
        c_out = self.c_out

        kx = s[3] - self.kernel_dims[0] + self.skip
        ky = s[4] - self.kernel_dims[1] + self.skip
        kz = s[5] - self.kernel_dims[2] + self.skip

        z = torch.zeros([batch, c_out, kx, ky, kz], dtype=x.dtype)

        for i in range(kx):
            print(i)
            for j in range(ky):
                for k in range(kz):
                    xx = x[:, :, :, i: i + self.kernel_dims[0], j: j + self.kernel_dims[1], k: k + self.kernel_dims[2]]

                    xxx = xx.reshape(batch, -1).type(self.weight.dtype)
                    z[:, :, i, j, k] = torch.mm(self.weight.view(c_out, -1), xxx.t()).t()

        return z
