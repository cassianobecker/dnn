from __future__ import print_function

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from dataset.mnist.torch_data import loaders
from experiments.mnist.runner import Runner
from util.experiment import get_experiment_params
from util.torch import seed_everything


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

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

    train_loader, test_loader = loaders(2000, 1000, 50)

    model = Net()

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
