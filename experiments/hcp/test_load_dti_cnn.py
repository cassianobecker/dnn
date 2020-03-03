from __future__ import print_function

import os
import argparse

import torch
import torch.utils.data
import torch.nn.functional as functional

from util.torch import seed_everything
from util.experiment import get_experiment_params

from dataset.hcp.torch_data import HcpDataset, HcpDataLoader


def experiment(params, args):
    """
    Sets up the experiment environment (loggers, data loaders, model, optimizer and scheduler) and initiates the
    train/test process for the model.
    :param args: keyword arguments from main() as parameters for the experiment
    :return: None
    """

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_set = HcpDataset(params, device, 'train')
    train_loader = HcpDataLoader(train_set, shuffle=False, batch_size=2)

    train_set.data_for_subject('100307')

    for batch_idx, (dti_tensors, targets, subjects) in enumerate(train_loader):

        print(batch_idx)
        print(dti_tensors.shape)
        print(targets.shape)


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