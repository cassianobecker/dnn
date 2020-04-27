import torch
import torch.nn as nn
import torch.nn.functional as F


class DtiConv3dTorch(nn.Module):
    def __init__(self, out_channels, kernel_size, stride):
        super(DtiConv3dTorch, self).__init__()

        self.stride = stride
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dti_dim = 3

        self.weight = nn.Parameter(torch.Tensor(out_channels, self.dti_dim, self.dti_dim,
                                                kernel_size[0], kernel_size[1], kernel_size[2]))
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


class DtiConv3dTorchVect(nn.Module):
    def __init__(self, out_channels, kernel_size, stride):
        super(DtiConv3dTorchVect, self).__init__()

        self.stride = stride
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dti_dim = 9

        self.weight = nn.Parameter(torch.Tensor(out_channels, self.dti_dim,
                                                kernel_size[0], kernel_size[1], kernel_size[2]))
        self.register_parameter('weight', self.weight)
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):

        y = F.conv3d(x.type(self.weight.dtype), self.weight, stride=self.stride)

        return y


class DtiConv3dTorchVectFirst(nn.Module):
    def __init__(self, out_channels):
        super(DtiConv3dTorchVectFirst, self).__init__()

        self.stride = 1
        self.out_channels = out_channels
        self.kernel_size = 1
        self.dti_dim = 9

        self.weight = nn.Parameter(torch.Tensor(self.dti_dim, out_channels))
        self.register_parameter('weight', self.weight)
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):

        y = torch.matmul(x.permute(2, 3, 4, 0, 1), self.weight).permute(3, 4, 0, 1, 2)

        return y


class DtiConv3d(nn.Module):
    def __init__(self, out_channels, kernel_size, skip):
        super(DtiConv3d, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.skip = skip
        self.dti_dim = 3

        self.weight = nn.Parameter(torch.Tensor(out_channels, self.dti_dim, self.dti_dim,
                                                kernel_size[0], kernel_size[1], kernel_size[2]))
        self.register_parameter('weight', self.weight)
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):

        xu = x.unfold(3, self.kernel_size[0], 1).unfold(4, self.kernel_size[1], 1).unfold(5, self.kernel_size[2], 1) \
            .permute(3, 4, 5, 0, 1, 2, 6, 7, 8)

        y = torch.einsum('cmnijk,stulmnijk->lcstu', self.weight, xu)

        return y


class DtiConv3dBatch(nn.Module):
    def __init__(self, c_out, kernel_dims, skip):
        super(DtiConv3dBatch, self).__init__()

        self.c_out = c_out
        self.kernel_dims = kernel_dims
        self.skip = skip
        self.dti_dim = 3

        self.weight = nn.Parameter(torch.Tensor(c_out, self.dti_dim, self.dti_dim,
                                                kernel_dims[0], kernel_dims[1], kernel_dims[2]))
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


class PyConv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_width, skip):
        super(PyConv2d, self).__init__()

        self.c_in  = c_in
        self.c_out = c_out
        self.kernel_width = kernel_width
        self.skip = skip

        self.weight = nn.Parameter(torch.Tensor(c_out, c_in, kernel_width, kernel_width))
        self.register_parameter('weight', self.weight)
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):

        xu = x.unfold(2, self.kernel_width, 1).unfold(3, self.kernel_width, 1)
        y = torch.einsum('lcstmn,ycmn->lyst', xu, self.weight)
        return y


def test_DtiConv3d():

    data_length = 8
    lift_dim = 6
    c_out1 = 32
    c_out2 = 20

    kernel_dims = [3, 3, 2]
    img_dim = [28, 28, lift_dim]
    dti_dim = 3
    skip = 1

    pool_width = 2
    linear_size1 = 5760
    linear_size2 = 128
    number_of_classes = 10

    conv1 = DtiConv3d(c_out1, kernel_dims, skip)
    conv2 = nn.Conv3d(c_out1, c_out2, kernel_dims, skip)

    dropout1 = nn.Dropout3d(0.25)
    dropout2 = nn.Dropout2d(0.5)

    fc1 = nn.Linear(linear_size1, linear_size2)
    fc2 = nn.Linear(linear_size2, number_of_classes)

    x0 = torch.rand(data_length, dti_dim, dti_dim, img_dim[0], img_dim[1], img_dim[2])

    x = conv1(x0)
    x = F.relu(x)
    x = conv2(x)
    x = F.max_pool3d(x, pool_width)
    x = dropout1(x)
    x = torch.flatten(x, 1)
    x = fc1(x)
    x = F.relu(x)
    x = dropout2(x)
    x = fc2(x)

    y = F.log_softmax(x, dim=1)

    pass


def test_PyConv2d():

    l = 64
    c_in = 1
    c_out = 32
    kernel_width = 3
    skip = 1
    h = 28

    conv1 = nn.Conv2d(c_in, c_out, kernel_width, skip)
    new_conv1 = PyConv2d(c_in, c_out, kernel_width, skip)

    W = conv1.weight
    new_conv1.weight = W
    x = torch.rand(l, c_in, h, h)

    y = conv1(x)
    y_new = new_conv1(x)

    pass


def test_conv():

    from skimage.util.shape import view_as_windows
    import numpy as np

    # batch length
    l = 12

    # data size
    w = 28
    h = 28
    d = 10

    # dti size
    m = 3
    n = 3

    # channels
    c = 5

    # kernel cube sizes
    i = 2
    j = 2
    k = 2

    q = np.zeros([c, m, n, i, j, k])

    x = np.zeros([l, m, n, w, h, d])
    xs = np.squeeze(view_as_windows(x, [l, m, n, i, j, k]))

    y = np.einsum('cmnijk,stulmnijk->lcstu', q, xs)

    xt = torch.tensor(x)
    qt = torch.tensor(q)

    xts = xt.unfold(3, i, 1).unfold(4, i, 1).unfold(5, i, 1)\
        .permute(3, 4, 5, 0, 1, 2, 6, 7, 8)

    print(np.sum(xs - xts.numpy()))

    yt = torch.einsum('cmnijk,stulmnijk->lcstu', qt, xts)

    print(np.sum(xs - xts.numpy()))

    pass


if __name__ == '__main__':
    test_DtiConv3d()
    test_PyConv2d()
    test_conv()
