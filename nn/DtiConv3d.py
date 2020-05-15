import torch
import torch.nn as nn
import torch.nn.functional as F


class DwiConv3dUnitKernel(nn.Module):
    def __init__(self, in_channels, out_channels, half_precision=False, cholesky_weights=False):
        super(DwiConv3dUnitKernel, self).__init__()

        if cholesky_weights is True and in_channels > 6:
            raise RuntimeError('Cholesky weights only valid if number of image channels is 6 (DTI Tensor).')

        self.out_channels = out_channels
        self.dti_dim = in_channels

        self.weight = nn.Parameter(torch.Tensor(self.dti_dim, out_channels))
        self.register_parameter('weight', self.weight)
        self.weight.data.uniform_(-0.1, 0.1)
        self.cholesky = cholesky_weights

        # required to implement Conv3d and layer size calculations
        self.kernel_size = 1
        self.stride = 1

    def _cholesky_weights(self):

        lifted_weight = torch.empty_like(self.weight)

        # FSL Convention:
        # 0: Dxx - l11
        # 1: Dxy - l12, l21
        # 2: Dxz - l13, l31
        # 3: Dyy - l22
        # 4: Dyz - l23, l32
        # 5: Dzz - l33

        # Dxx: l11 ^ 2
        lifted_weight[0, :] = self.weight[0, :] ** 2

        # Dxy: 2 * l21 * l11
        lifted_weight[1, :] = 2 * self.weight[1, :] * self.weight[0, :]

        # Dxz: 2 * l31 * l21
        lifted_weight[2, :] = 2 * self.weight[2, :] * self.weight[1, :]

        # Dyy: l21 ^ 2 +  l22 ^ 2
        lifted_weight[3, :] = self.weight[1, :] ** 2 + self.weight[3, :] ** 2

        # Dyz: 2 * (l31 * l21  +  l32 * l22)
        lifted_weight[4, :] = 2 * (self.weight[2, :] * self.weight[1, :] + self.weight[4, :] * self.weight[3, :])

        # Dzz: l13 ^ 2 + l23 ^ 2 + l33 ^ 2
        lifted_weight[5, :] = self.weight[2, :] ** 2  + self.weight[4, :] ** 2 + self.weight[5, :] ** 2

        return lifted_weight

    def forward(self, x):

        weight = self._cholesky_weights() if self.cholesky is True else self.weight

        y = torch.matmul(x.permute(2, 3, 4, 0, 1), weight).permute(3, 4, 0, 1, 2)

        return y


# class DwiConv3dTorch(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, half_precision=False):
#         super(DwiConv3dTorch, self).__init__()
#
#         self.stride = stride
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.dwi_dim = in_channels
#
#         self.weight = nn.Parameter(torch.Tensor(out_channels, self.dwi_dim, self.dwi_dim,
#                                                 kernel_size[0], kernel_size[1], kernel_size[2]))
#         if half_precision is True:
#             self.weight.half()
#
#         self.register_parameter('weight', self.weight)
#         self.weight.data.uniform_(-0.1, 0.1)
#
#     def forward(self, x):
#
#         sx = x.shape
#         sw = self.weight.shape
#
#         y = F.conv3d(x.view(sx[0], sx[1] * sx[2], sx[3], sx[4], sx[5]).type(self.weight.dtype),
#                      self.weight.view(sw[0], sw[1] * sw[2], sw[3], sw[4], sw[5]), stride=self.stride)
#
#         return y
#
#     def forward2(self, x):
#
#         xu = x.unfold(3, self.kernel_size[0], 1).unfold(4, self.kernel_size[1], 1).unfold(5, self.kernel_size[2], 1) \
#             .permute(3, 4, 5, 0, 1, 2, 6, 7, 8)
#
#         y = torch.einsum('cmnijk,stulmnijk->lcstu', self.weight, xu)
#
#         return y
#
#
# class DwiConv3dTorchVect(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, half_precision=False):
#         super(DwiConv3dTorchVect, self).__init__()
#
#         self.stride = stride
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.dwi_dim = in_channels
#
#         self.weight = nn.Parameter(torch.Tensor(out_channels, self.dwi_dim,
#                                                 kernel_size[0], kernel_size[1], kernel_size[2]))
#         if half_precision is True:
#             self.weight.half()
#
#         self.register_parameter('weight', self.weight)
#         self.weight.data.uniform_(-0.1, 0.1)
#
#     def forward(self, x):
#
#         y = F.conv3d(x.type(self.weight.dtype), self.weight, stride=self.stride)
#
#         return y
