from __future__ import print_function
import argparse
import os
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from fwk import learner
from ext.grid_graph import grid_graph
from dataset.mnist.torch_data import loaders
from nn.chebnet import ChebConv


class Net(torch.nn.Module):

    def __init__(self, adj, device, k, g):
        # f: number of input filters
        # g: number of output filters
        # k: order of Chebyshev polynomial
        # c: number of classes
        # n: number of vertices at coarsening level
        super(Net, self).__init__()

        self.edge_index = torch.tensor([adj.tocoo().row, adj.tocoo().col], dtype=torch.long).to(device)

        f1, g1, k1 = 1, g, k
        self.conv1 = ChebConv(f1, g1, K=k1)

        n1 = adj.shape[0]
        d = 10
        self.fc1 = torch.nn.Linear(n1 * g1, d)

    def forward(self, x):
        x = self.class_scores(x);
        return F.log_softmax(x, dim=0)

    def class_scores(self, x):
        x = self.conv1(x, self.edge_index)
        x = F.relu(x)

        x = x.view(-1)
        x = self.fc1(x)

        return x


def experiment(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_size = 2000
    test_size = 1000

    train_loader, test_loader = loaders(train_size, test_size, args.batch_size)

    shuffles = [False]
    ks = [5, 15]
    gs = [10, 30]
    es = [4, 8]

    for e in es:
        for k in ks:
            for g in gs:
                for shuffle in shuffles:

                    print('==========================================')
                    print('edges={:d} | k={:d} | g={:d} | shuffle={:}'
                          .format(e, k, g, shuffle))
                    print('==========================================')

                    adj = grid_graph(28, number_edges=e, corners=False, shuffled=shuffle)

                    model = Net(adj, device, k, g)
                    model.to(device)

                    optimizer = optim.Adam(model.parameters(), lr=args.lr)
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

                    for epoch in range(1, args.epochs):

                        learner.train(args, model, device, train_loader, optimizer, epoch)
                        scheduler.step()

                        learner.test(args, model, device, test_loader, epoch)

                        if args.save_model:
                            save_path = os.path.join('out', 'models', f'mnist_gcn_A{e}_k{k}_g{g}_epoch{epoch}.pt')
                            print('Saving model to: {:}.'.format(save_path))
                            torch.save(model.state_dict(), save_path)

                    print('\n--------- FINISHED EXPERIMENT ------')


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
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

    experiment(args)


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    seed_everything(1234)
    main()
