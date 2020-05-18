import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from fwk.config import Config
from fwk.metrics import MetricsHandler
from fwk.model import ModelHandler
from util.lang import to_bool

from dataset.mnist.loader import MnistDataset, MnistDataLoader
from dataset.mnist.images import Images
from util.lang import class_for_name
from experiments.mnist.architectures_simple import Net


class BatchTrain:

    def __init__(self):
        self.epochs = None
        self.model = None
        self.loss = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.data_loaders = dict()
        self.accumulation_steps = None

    def execute(self):

        MetricsHandler.dispatch_event(locals(), 'before_setup')

        self.setup()

        MetricsHandler.dispatch_event(locals(), 'after_setup')

        if to_bool(Config.config['OUTPUTS']['load_model']) is True:
            self.model = ModelHandler.load_model(epoch=1)

        for epoch in range(self.epochs):

            MetricsHandler.dispatch_event(locals(), 'before_epoch')

            self.train_batch(epoch)
            self.test_batch(epoch)

            MetricsHandler.dispatch_event(locals(), 'after_epoch')

            if to_bool(Config.config['OUTPUTS']['save_model']) is True:
                ModelHandler.save_model(self.model, epoch)

        self._teardown()

    def _teardown(self):
        pass

    def setup(self):

        torch.manual_seed(1234)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # num_classes = 10

        # half_precision = to_bool(Config.get_option('ALGORITHM', 'half_precision', 'False'))
        # max_img_channels = int(Config.get_option('ALGORITHM', 'max_img_channels', 1000))
        # cholesky_weights = to_bool(Config.get_option('ARCHITECTURE', 'cholesky_weights', 'False'))

        # train_images, test_images = Images.create_list_from_config()
        #
        # train_set = MnistDataset(
        #     self.device,
        #     images=train_images,
        #     regime='train',
        #     half_precision=half_precision,
        #     max_img_channels=max_img_channels
        # )
        #
        # self.data_loaders['train'] = MnistDataLoader(
        #     train_set,
        #     shuffle=False,
        #     batch_size=int(Config.config['ALGORITHM']['train_batch_size'])
        # )
        #
        # test_set = MnistDataset(
        #     self.device,
        #     images=test_images,
        #     regime='test',
        #     half_precision=half_precision,
        #     max_img_channels=max_img_channels
        # )
        #
        # self.data_loaders['test'] = MnistDataLoader(
        #     test_set,
        #     shuffle=False,
        #     batch_size=int(Config.config['ALGORITHM']['test_batch_size'])
        # )

        # device = torch.device("cuda" if use_cuda else "cpu")
        # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        kwargs = {}

        train_batch_size = int(Config.config['ALGORITHM']['train_batch_size'])
        test_batch_size = int(Config.config['ALGORITHM']['train_batch_size'])

        self.data_loaders['train'] = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=train_batch_size, shuffle=True, **kwargs)

        self.data_loaders['test'] = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)

        # arch_class_name = Config.config['ARCHITECTURE']['arch_class_name']
        # model_class = class_for_name(arch_class_name)
        # self.model = model_class(img_dims, num_classes, cholesky_weights=cholesky_weights)
        # self.model.to(self.device)

        self.model = Net().to(self.device)

        self.epochs = int(Config.config['ALGORITHM']['epochs'])

        self.optimizer = optim.Adadelta(
            self.model.parameters(),
            lr=float(Config.config['ALGORITHM']['lr'])
        )

        self.scheduler = StepLR(
            self.optimizer,
            step_size=1,
            gamma=float(Config.config['ALGORITHM']['gamma'])
        )

        if Config.config.has_option('ALGORITHM', 'accumulation_steps'):
            self.accumulation_steps = int(Config.config['ALGORITHM']['test_batch_size'])
        else:
            self.accumulation_steps = 1

    def train_batch(self, epoch):

        self.model.train()

        # for batch_idx, (dti_tensors, targets, subjects) in enumerate(self.data_loaders['train']):
        for batch_idx, (dti_tensors, targets) in enumerate(self.data_loaders['train']):

            MetricsHandler.dispatch_event(locals(), 'before_train_batch')

            dti_tensors, targets = dti_tensors.to(self.device).type(torch.float32), \
                                   targets.to(self.device).type(torch.long)

            self.optimizer.zero_grad()

            outputs = self.model(dti_tensors)

            loss = F.nll_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # if (batch_idx + 1) % self.accumulation_steps == 0:
            #     self.optimizer.step()
            #     self.model.zero_grad()

            # begin from simple
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(dti_tensors), len(self.data_loaders['train'].dataset),
                    100. * batch_idx / len(self.data_loaders['train']), loss.item()))
            # end from simple

            MetricsHandler.dispatch_event(locals(), 'after_train_batch')

    # def train_batch(self, epoch):
    #     self.model.train()
    #     for batch_idx, (data, targets) in enumerate(self.data_loaders['train']):
    #         data, targets = data.to(self.device), targets.to(self.device)
    #         self.optimizer.zero_grad()
    #         outputs = self.model(data)
    #         loss = F.nll_loss(outputs, targets)
    #         loss.backward()
    #         self.optimizer.step()
    #         if batch_idx % 10 == 0:
    #             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                 epoch, batch_idx * len(data), len(self.data_loaders['train'].dataset),
    #                 100. * batch_idx / len(self.data_loaders['train']), loss.item()))
    #
    #         MetricsHandler.dispatch_event(locals(), 'after_train_batch')

    def test_batch(self, epoch):

        self.model.eval()

        # begin from simple
        test_loss = 0
        correct = 0
        # end from simple

        with torch.no_grad():

            # for batch_idx, (dti_tensors, targets, subjects) in enumerate(self.data_loaders['test']):
            for batch_idx, (dti_tensors, targets) in enumerate(self.data_loaders['test']):

                MetricsHandler.dispatch_event(locals(), 'before_test_batch')

                dti_tensors, targets = dti_tensors.to(self.device).type(torch.float32),\
                                       targets.to(self.device).type(torch.long)

                outputs = self.model(dti_tensors)

                # begin from simple
                test_loss += F.nll_loss(outputs, targets, reduction='sum').item()  # sum up batch loss
                pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(targets.view_as(pred)).sum().item()
                # end from simple

                MetricsHandler.dispatch_event(locals(), 'after_test_batch')

        # begin from simple
        test_loss /= len(self.data_loaders['test'].dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.data_loaders['test'].dataset),
            100. * correct / len(self.data_loaders['test'].dataset)))
        # end from simple

    # def test_batch(self, epoch):
    #     self.model.eval()
    #     test_loss = 0
    #     correct = 0
    #     with torch.no_grad():
    #         for data, targets in self.data_loaders['test']:
    #             data, targets = data.to(self.device), targets.to(self.device)
    #             outputs = self.model(data)
    #             test_loss += F.nll_loss(outputs, targets, reduction='sum').item()  # sum up batch loss
    #             pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #             correct += pred.eq(targets.view_as(pred)).sum().item()
    #
    #             MetricsHandler.dispatch_event(locals(), 'after_test_batch')
    #
    #     test_loss /= len(self.data_loaders['test'].dataset)
    #
    #     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         test_loss, correct, len(self.data_loaders['test'].dataset),
    #         100. * correct / len(self.data_loaders['test'].dataset)))
    #
    #     pass
