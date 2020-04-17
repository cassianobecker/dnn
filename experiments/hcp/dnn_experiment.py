import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from fwk.config import Config
from fwk.metrics import MetricsHandler
from fwk.model import ModelHandler
from util.lang import to_bool

from dataset.hcp.torch_data import HcpDataset, HcpDataLoader
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

    def execute(self):

        MetricsHandler.collect_metrics(locals(), 'BEFORE_SETUP')
        self.setup()
        MetricsHandler.collect_metrics(locals(), 'AFTER_SETUP')

        if to_bool(Config.config['OUTPUTS']['load_model']) is True:
            self.model = ModelHandler.load_model(epoch=1)

        for epoch in range(self.epochs):

            MetricsHandler.collect_metrics(locals(), 'BEFORE_EPOCH')
            self.train_batch(epoch)
            self.test_batch(epoch)
            MetricsHandler.collect_metrics(locals(), 'AFTER_EPOCH')

            if to_bool(Config.config['OUTPUTS']['save_model']) is True:
                ModelHandler.save_model(self.model, epoch)

        MetricsHandler.collect_metrics(locals(), 'BEFORE_TEARDOWN')
        self._teardown()
        MetricsHandler.collect_metrics(locals(), 'AFTER_TEARDOWN')

    def _teardown(self):
        pass

    def setup(self):

        torch.manual_seed(1234)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_set = HcpDataset(self.device, 'train')
        self.data_loaders['train'] = HcpDataLoader(
            train_set,
            shuffle=False,
            batch_size=int(Config.config['ALGORITHM']['train_batch_size'])
        )

        test_set = HcpDataset(self.device, 'test')
        self.data_loaders['test'] = HcpDataLoader(
            test_set,
            shuffle=False,
            batch_size=int(Config.config['ALGORITHM']['test_batch_size'])
        )

        arch_class_name = Config.config['ARCHITECTURE']['arch_class_name']
        model_class = class_for_name(arch_class_name)
        self.model = model_class().to(self.device)

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

    def train_batch(self, epoch):

        self.model.train()

        for batch_idx, (dti_tensors, targets, subjects) in enumerate(self.data_loaders['train']):

            MetricsHandler.collect_metrics(locals(), 'BEFORE_TRAIN_BATCH')

            dti_tensors, targets = dti_tensors.to(self.device), targets.to(self.device).type(torch.long)

            self.optimizer.zero_grad()
            outputs = self.model(dti_tensors)
            loss = F.nll_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            MetricsHandler.collect_metrics(locals(), 'AFTER_TRAIN_BATCH')

    def test_batch(self, epoch):

        self.model.eval()

        with torch.no_grad():

            for batch_idx, (dti_tensors, targets, subjects) in enumerate(self.data_loaders['test']):

                MetricsHandler.collect_metrics(locals(), 'BEFORE_TEST_BATCH')

                dti_tensors, targets = dti_tensors.to(self.device), targets.to(self.device).type(torch.long)

                outputs = self.model(dti_tensors)

                MetricsHandler.collect_metrics(locals(), 'AFTER_TEST_BATCH')
