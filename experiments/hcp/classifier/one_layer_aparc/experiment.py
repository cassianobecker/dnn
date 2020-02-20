from __future__ import print_function

import os
import argparse

import torch
import torch.utils.data
import torch.nn.functional as functional

from util.torch import seed_everything
from util.experiment import get_experiment_params

from dataset.hcp.torch_data import HcpDataset, HcpDataLoader

from nn.chebnet import ChebTimeConvDeprecated

from experiments.hcp.classifier.runner import Runner


class NetTGCNBasic(torch.nn.Module):
    """
    A 1-Layer time graph convolutional network
    :param mat_size: temporary parameter to fix the FC1 size
    """

    def __init__(self, mat_size):
        super(NetTGCNBasic, self).__init__()

        f1, g1, k1, h1 = 1, 64, 25, 15
        self.conv1 = ChebTimeConvDeprecated(f1, g1, K=k1, H=h1)

        n2 = mat_size
        c = 6
        self.fc1 = torch.nn.Linear(int(n2 * g1), c)

    def forward(self, x, graph_list, mapping_list):
        """
        Computes forward pass through the time graph convolutional network
        :param x: windowed BOLD signal to as input to the TGCN
        :return: output of the TGCN forward pass
        """

        x = self.conv1(x, graph_list[0][0])

        x = functional.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return functional.log_softmax(x, dim=1)

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def experiment(params, args):
    """
    Sets up the experiment environment (loggers, data loaders, model, optimizer and scheduler) and initiates the
    train/test process for the model.
    :param args: keyword arguments from main() as parameters for the experiment
    :return: None
    """

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    coarsen = None

    train_set = HcpDataset(params, device, 'train', coarsen=coarsen)
    train_loader = HcpDataLoader(train_set, shuffle=False)

    test_set = HcpDataset(params, device, 'test', coarsen=coarsen)
    test_loader = HcpDataLoader(test_set, shuffle=False)

    data_shape = train_set.data_shape()

    model = NetTGCNBasic(data_shape)

    runner = Runner(device, params, train_loader, test_loader)

    model = runner.initial_save_and_load(model, restart=False)

    runner.run(args, model, run_initial_test=False)


if __name__ == '__main__':
    params = get_experiment_params(__file__, __name__)

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    seed_everything(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description=__name__)

    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    args.reg_weight = 0.

    experiment(params, args)
