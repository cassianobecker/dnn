import torch
import torch.nn as nn
import torch.nn.functional as F


class DtiConv3d(nn.Module):
    def __init__(self, c_out, kernel_dims, skip):
        super(DtiConv3d, self).__init__()

        self.c_out = c_out
        self.kernel_dims = kernel_dims
        self.skip = skip
        self.dti_dim = 3

        self.weight = nn.Parameter(torch.Tensor(c_out, self.dti_dim, self.dti_dim,
                                                kernel_dims[0], kernel_dims[1], kernel_dims[2]))
        self.register_parameter('weight', self.weight)
        self.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):

        xu = x.unfold(3, self.kernel_dims[0], 1).unfold(4, self.kernel_dims[1], 1).unfold(5, self.kernel_dims[2], 1) \
            .permute(3, 4, 5, 0, 1, 2, 6, 7, 8)

        y = torch.einsum('cmnijk,stulmnijk->lcstu', self.weight, xu)

        return y


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
