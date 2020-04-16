from __future__ import print_function
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataset.hcp.torch_data import HcpDataset, HcpDataLoader
from util.lang import class_for_name
from fwk.config import Config, ConfigProductGenerator
from util.path import append_path
from fwk.metrics import MetricsHandler


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

        MetricsHandler.collect_metrics(locals(), 'after_batch')

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


class Args:

    def __init__(self):
        self.batch_size = int(Config.config['ALGORITHM']['batch_size'])
        self.test_batch_size = int(Config.config['ALGORITHM']['test_batch_size'])
        self.epochs = int(Config.config['ALGORITHM']['epochs'])
        self.lr = float(Config.config['ALGORITHM']['lr'])
        self.gamma = float(Config.config['ALGORITHM']['gamma'])


def main():

    MetricsHandler.add_metric('util.torch_metrics.GradientMetrics', 'after_batch')
    MetricsHandler.add_metric('util.torch_metrics.SubjectMetrics', 'before_experiment')

    args = Args()

    torch.manual_seed(1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = HcpDataset(device, 'train')
    train_loader = HcpDataLoader(train_set, shuffle=False, batch_size=args.batch_size)

    test_set = HcpDataset(device, 'test')
    test_loader = HcpDataLoader(test_set, shuffle=False, batch_size=args.test_batch_size)

    arch_class_name = Config.config['ARCHITECTURE']['arch_class_name']
    model_class = class_for_name(arch_class_name)
    model = model_class().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    MetricsHandler.collect_metrics(locals(), 'before_experiment')


    for epoch in range(1, args.epochs + 1):

        train(args, model, device, train_loader, optimizer, epoch)

        if args.save_model:
            torch.save(model.state_dict(), "hcp_dti_cnn.pt")

        test(args, model, device, test_loader)

        scheduler.step()


def initialize_config(argv):

    if len(sys.argv) == 1:

        config_url = append_path(__file__, 'conf/args.ini')
        config_generator = ConfigProductGenerator(config_url)

        if config_generator.has_multiple_products():
            print('Multiple configurations available, picking first')

        Config.set_config(config_generator.config_products[0])

    else:
        Config.set_config(sys.argv[1])


if __name__ == '__main__':

    initialize_config(sys.argv)

    main()
