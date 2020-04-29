from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from nn.DtiConv3d import DwiConv3dTorch
from metrics.dataset import batch_tensor_to_3dti


class Net(nn.Module):

    def __init__(self, lift_dim=6):
        super(Net, self).__init__()

        self.lift_dim = lift_dim

        img_dims = [28, 28, lift_dim]
        kernel_dims = [5, 6, 2]

        c_out1 = 32
        c_out2 = 64
        strides = 1

        self.pool_width = 2

        linear_size1 = (c_out2 / (self.pool_width ** 3)) *\
                       (img_dims[0] - kernel_dims[0] - 1) *\
                       (img_dims[1] - kernel_dims[1] - 1) *\
                       (img_dims[2] - kernel_dims[2])

        linear_size1 = 11520
        linear_size2 = 128
        number_of_classes = 10

        self.conv1 = DwiConv3dTorch(c_out1, kernel_dims, strides)
        self.conv2 = nn.Conv3d(c_out1, c_out2, kernel_dims, 1)

        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool3d(x, self.pool_width)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        data_dti = batch_tensor_to_3dti(data, lift_dim=args.lift_dim).to(device)

        optimizer.zero_grad()

        output = model(data_dti)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            data_dti = batch_tensor_to_3dti(data, lift_dim=args.lift_dim).to(device)

            output = model(data_dti)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():

    lift_dim = 6
    batch_size = 16
    test_batch_size = 1000

    train_dataset_size = 5000
    test_dataset_size = 2000

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

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

    parser.add_argument('--lift_dim', type=int, default=lift_dim, metavar='N',
                        help='dimension to 3d lift of dti tensor')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    sub_idx_train = list(range(0, train_dataset_size))
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    train_set_sub = torch.utils.data.Subset(train_set, sub_idx_train)
    train_loader = torch.utils.data.DataLoader(train_set_sub, batch_size=args.batch_size, shuffle=True, **kwargs)

    sub_idx_test = list(range(0, test_dataset_size))
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_set = datasets.MNIST('../data', train=False, transform=test_transforms)
    test_set_sub = torch.utils.data.Subset(test_set, sub_idx_test)
    test_loader = torch.utils.data.DataLoader(test_set_sub, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net(lift_dim=args.lift_dim).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_dti_cnn.pt")


if __name__ == '__main__':
    main()
