import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from fwk.config import Config
from fwk.metrics import MetricsHandler
from fwk.model import ModelHandler
from util.lang import to_bool

from dataset.hcp.loader import HcpDataset, HcpDataLoader
from dataset.hcp.subjects import Subjects
from util.lang import class_for_name
from util.encode import one_hot_to_int


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
        self.regression = None

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

            self.scheduler.step()

            MetricsHandler.dispatch_event(locals(), 'after_epoch')

            if to_bool(Config.config['OUTPUTS']['save_model']) is True:
                ModelHandler.save_model(self.model, epoch)

        self._teardown()

    def _teardown(self):
        pass

    def setup(self):

        torch.manual_seed(1234)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        half_precision = to_bool(Config.get_option('ALGORITHM', 'half_precision', 'False'))
        max_img_channels = int(Config.get_option('ALGORITHM', 'max_img_channels', 1000))
        cholesky_weights = to_bool(Config.get_option('ARCHITECTURE', 'cholesky_weights', 'False'))

        perturb = to_bool(Config.get_option('DATABASE', 'perturb', 'False'))
        self.regression = to_bool(Config.get_option('COVARIATES', 'regression', 'False'))

        train_subjects, test_subjects = Subjects.create_list_from_config()

        train_set = HcpDataset(
            self.device,
            subjects=train_subjects,
            half_precision=half_precision,
            max_img_channels=max_img_channels,
            perturb=perturb,
            regression=self.regression
        )

        self.data_loaders['train'] = HcpDataLoader(
            train_set,
            shuffle=False,
            batch_size=int(Config.config['ALGORITHM']['train_batch_size'])
        )

        test_set = HcpDataset(
            self.device,
            subjects=test_subjects,
            half_precision=half_precision,
            max_img_channels=max_img_channels,
            perturb=False,
            regression=self.regression
        )

        self.data_loaders['test'] = HcpDataLoader(
            test_set,
            shuffle=False,
            batch_size=int(Config.config['ALGORITHM']['test_batch_size'])
        )

        img_dims = train_set.tensor_size()
        num_classes = train_set.number_of_classes()

        arch_class_name = Config.config['ARCHITECTURE']['arch_class_name']
        model_class = class_for_name(arch_class_name)

        self.model = model_class(img_dims, num_classes, cholesky_weights=cholesky_weights)
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

        self.accumulation_steps = int(Config.get_option('ALGORITHM', 'accumulation_steps', 1))

    def train_batch(self, epoch):

        self.model.train()

        self.optimizer.zero_grad()

        for batch_idx, (dwi_tensors, targets, subjects) in enumerate(self.data_loaders['train']):

            MetricsHandler.dispatch_event(locals(), 'before_train_batch')

            # dwi_tensors, targets = dwi_tensors.to(self.device).type(
            #     torch.float32), targets.to(self.device).type(torch.long)

            dwi_tensors, targets = dwi_tensors.to(self.device).type(
                torch.float32), targets.to(self.device).type(torch.float32)

            # self.optimizer.zero_grad()
            outputs = self.model(dwi_tensors)

            if self.regression is True:
                loss = F.mse_loss(outputs, targets)
            else:
                loss = F.nll_loss(outputs, one_hot_to_int(targets))

            loss.backward()
            # self.optimizer.step()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.model.zero_grad()

            MetricsHandler.dispatch_event(locals(), 'after_train_batch')

    def test_batch(self, epoch):

        self.model.eval()

        with torch.no_grad():
            for batch_idx, (dwi_tensors, targets, subjects) in enumerate(self.data_loaders['test']):
                MetricsHandler.dispatch_event(locals(), 'before_test_batch')

                dwi_tensors, targets = dwi_tensors.to(self.device).type(
                    torch.float32), targets.to(self.device).type(torch.long)

                outputs = self.model(dwi_tensors)

                MetricsHandler.dispatch_event(locals(), 'after_test_batch')
