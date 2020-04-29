import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from fwk.config import Config
from fwk.metrics import MetricsHandler
from fwk.model import ModelHandler
from util.lang import to_bool

from dataset.hcp.loader import HcpDataset, HcpDataLoader
from util.lang import class_for_name


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

        num_classes = 2

        torch.manual_seed(1234)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if Config.config.has_option('ALGORITHM', 'half_precision'):
            half_precision = to_bool(Config.config['ALGORITHM']['half_precision'])
        else:
            half_precision = False

        if Config.config.has_option('ALGORITHM', 'max_img_channels'):
            max_img_channels = int(Config.config['ALGORITHM']['max_img_channels'])
        else:
            max_img_channels = None

        train_set = HcpDataset(
            self.device,
            regime='train',
            half_precision=half_precision,
            max_img_channels=max_img_channels
        )

        self.data_loaders['train'] = HcpDataLoader(
            train_set,
            shuffle=False,
            batch_size=int(Config.config['ALGORITHM']['train_batch_size'])
        )

        test_set = HcpDataset(
            self.device,
            regime='test',
            half_precision=half_precision,
            max_img_channels=max_img_channels
        )

        self.data_loaders['test'] = HcpDataLoader(
            test_set,
            shuffle=False,
            batch_size=int(Config.config['ALGORITHM']['test_batch_size'])
        )

        img_dims = train_set.tensor_size()

        arch_class_name = Config.config['ARCHITECTURE']['arch_class_name']
        model_class = class_for_name(arch_class_name)
        self.model = model_class(img_dims, num_classes, half_precision=half_precision)

        self.model.to(self.device)

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

        for batch_idx, (dti_tensors, targets, subjects) in enumerate(self.data_loaders['train']):

            MetricsHandler.dispatch_event(locals(), 'before_train_batch')

            dti_tensors, targets = dti_tensors.to(self.device).type(torch.float32), \
                                   targets.to(self.device).type(torch.long)

            self.optimizer.zero_grad()

            outputs = self.model(dti_tensors)

            loss = F.nll_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            MetricsHandler.dispatch_event(locals(), 'after_train_batch')

    def test_batch(self, epoch):

        self.model.eval()

        with torch.no_grad():

            for batch_idx, (dti_tensors, targets, subjects) in enumerate(self.data_loaders['test']):

                MetricsHandler.dispatch_event(locals(), 'before_test_batch')

                dti_tensors, targets = dti_tensors.to(self.device).type(torch.float32),\
                                       targets.to(self.device).type(torch.long)

                outputs = self.model(dti_tensors)

                MetricsHandler.dispatch_event(locals(), 'after_test_batch')
