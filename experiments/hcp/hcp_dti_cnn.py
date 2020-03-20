from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from nn.DtiConv3d import DtiConv3dTorch
from dataset.hcp.torch_data import HcpDataset, HcpDataLoader
from util.experiment import get_experiment_params


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        #img_dims = [28, 28, lift_dim]
        img_dims = [145, 174, 145]

        kernel_dims1 = [10, 10, 10]
        kernel_dims2 = [5, 5, 5]

        c_out1 = 2*4*10
        c_out2 = 3*4*10
        skip = 1

        self.pool_width = 2

        linear_size1 = (c_out2 / (self.pool_width ** 3)) *\
                       (img_dims[0] - kernel_dims1[0] - 1) *\
                       (img_dims[1] - kernel_dims1[1] - 1) *\
                       (img_dims[2] - kernel_dims1[2])

        linear_size1 = 513000
        # linear_size1 = 217800
        linear_size2 = 128
        number_of_classes = 2
        strides = [4, 4, 4]

        self.conv1 = DtiConv3dTorch(c_out1, kernel_dims1, strides)

        self.conv2 = nn.Conv3d(c_out1, c_out2, kernel_dims2, skip)

        self.conv3 = nn.Conv3d(c_out2, c_out2, kernel_dims2, skip)

        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # self.dropout3 = nn.Dropout2d(0.25)


        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = x * 3000
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool3d(x, self.pool_width)
        x = self.dropout1(x)
        # x = F.relu(x)
        # x = self.conv3(x)
        # x = self.dropout3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    train_loss = 0
    correct = 0

    for batch_idx, (dti_tensors, targets, subjects) in enumerate(train_loader):

        dti_tensors, targets = dti_tensors.to(device), targets.to(device).type(torch.long)

        optimizer.zero_grad()

        print('training on subject {}'.format(subjects))

        output = model(dti_tensors)
        loss = F.nll_loss(output, targets)
        # loss = F.nll_loss(output, targets.max(dim=0)[1])
        loss.backward()
        optimizer.step()

        train_loss += F.nll_loss(output, targets, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        # print(pred)
        # print(targets)
        # correct += pred.eq(targets.view_as(pred)).sum().item()
        correct += pred.eq(targets.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

    if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(dti_tensors), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (dti_tensors, targets, subjects) in enumerate(test_loader):

            dti_tensors, targets = dti_tensors.to(device), targets.to(device).type(torch.long)

            # print('testing on subject {}'.format(subjects))

            output = model(dti_tensors)

            # test_loss += F.nll_loss(output, targets.max(dim=0)[1], reduction='sum').item()  # sum up batch loss
            test_loss += F.nll_loss(output, targets, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            print(pred)
            print(targets)
            # correct += pred.eq(targets.view_as(pred)).sum().item()
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():

    batch_size = 2
    test_batch_size = 4

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch HCP Diffusion Example')

    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=test_batch_size, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    params = get_experiment_params(__file__, __name__)

    train_set = HcpDataset(params, device, 'train')
    train_loader = HcpDataLoader(train_set, shuffle=False, batch_size=batch_size)

    test_set = HcpDataset(params, device, 'test')
    test_loader = HcpDataLoader(test_set, shuffle=False, batch_size=test_batch_size)

    model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):

        train(args, model, device, train_loader, optimizer, epoch)

        if args.save_model:
            torch.save(model.state_dict(), "hcp_dti_cnn.pt")

        test(args, model, device, test_loader)

        scheduler.step()


if __name__ == '__main__':
    main()
