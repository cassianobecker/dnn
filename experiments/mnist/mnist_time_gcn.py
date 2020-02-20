from __future__ import print_function
import argparse
import os
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from ext.grid_graph import grid_graph
from dataset.mnist.torch_data import loaders
from nn.chebnet import ChebConvTimeFFT

horizon = 4
in_channels = 3


class TimeNet(torch.nn.Module):

    def __init__(self, adj, device, order_filter, out_channels, horizon):
        super(TimeNet, self).__init__()

        num_vertices = adj.shape[0]
        num_classes = 10

        self.edge_index = torch.tensor([adj.tocoo().row, adj.tocoo().col], dtype=torch.long).to(device)

        self.time_conv1 = ChebConvTimeFFT(in_channels, out_channels, order_filter=order_filter, horizon=horizon)

        self.fc1 = torch.nn.Linear(num_vertices * out_channels, num_classes)

    def forward(self, x):
        x = self.time_conv1(x, self.edge_index)
        x = F.relu(x)

        # average pool across time-dimension
        x = x.sum(dim=1)

        # fully-connected layer
        x = x.view(-1)
        x = self.fc1(x)

        return F.log_softmax(x, dim=0)


def train(args, model, device, train_loader, optimizer, epoch):
    torch.cuda.synchronize()
    model.train()
    for batch_idx, (data_t, target_t) in enumerate(train_loader):
        data = data_t.to(device)
        target = target_t.to(device)
        optimizer.zero_grad()
        N = data.shape[0]
        outputs = [model(data[i, :].repeat(in_channels, horizon, 1).permute(2, 1, 0)) for i in range(N)]
        targets = [target[i, :].argmax() for i in range(N)]
        loss = sum([F.nll_loss(outputs[i].unsqueeze(0), targets[i].unsqueeze(-1)) for i in range(N)])
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {:2d} [{:5d}/{:5d} ({:2.0f}%)] Loss: {:1.5e}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))

    torch.cuda.synchronize()


def test(_, model, device, test_loader, epoch):
    model.eval()
    sum_correct = 0.
    test_loss = 0.
    with torch.no_grad():
        for data_t, target_t in test_loader:
            data = data_t.to(device)
            target = target_t.to(device)
            N = data.shape[0]
            outputs = [model(data[i, :].repeat(in_channels, horizon, 1).permute(2, 1, 0)) for i in range(N)]
            preds = [outputs[i].argmax() for i in range(N)]
            targets = [target[i, :].argmax() for i in range(N)]
            correct = sum([targets[i] == preds[i] for i in range(N)]).item()
            sum_correct += correct
            # print(float(correct)/N)
            test_loss += sum([F.nll_loss(outputs[i].unsqueeze(0), targets[i].unsqueeze(-1)) for i in range(N)])

    test_loss /= len(test_loader.dataset)

    print('Epoch: {:3d}, AvgLoss: {:.4f}, Accuracy: {:.4f}'.format(
        epoch, test_loss, float(sum_correct) / len(test_loader.dataset)))


def experiment(args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_size = 2000
    test_size = 1000

    train_loader, test_loader = loaders(train_size, test_size, args.batch_size)

    shuffle = False
    order_filter = 15
    out_channels = 10
    number_edges = 8

    print('==========================================')
    print('-- Time Graph Convolution for MNIST ')
    print('edges={:d} | k={:d} | g={:d} | shuffle={:}'
          .format(number_edges, order_filter, out_channels, shuffle))
    print('==========================================')

    adj = grid_graph(28, number_edges=number_edges, corners=False, shuffled=shuffle)

    model = TimeNet(adj, device, order_filter, out_channels, horizon)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(1, args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(args, model, device, test_loader, epoch)

    if args.save_model:
        save_path = os.path.join('out', 'models', "mnist_time_gcn.pt")
        print('Saving model to: {:}.'.format(save_path))
        torch.save(model.state_dict(), save_path)


def main():
    # python - u  pygeo_mnist_time_basic.py > out.txt
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
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
    parser.add_argument('--save-model', action='store_true', default=False,
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
