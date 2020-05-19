import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from fwk.config import Config
from fwk.metrics import MetricsHandler
from fwk.model import ModelHandler
from util.lang import to_bool

from dataset.mnist.loader import MnistDataset, MnistDataLoader
from dataset.mnist.images import Images
from util.lang import class_for_name
from torch.autograd import Variable
from dipy.io.image import save_nifti
import os
import numpy as np
from torch.optim import Adam


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

        for epoch in range(self.epochs):

            MetricsHandler.dispatch_event(locals(), 'before_epoch')

            MetricsHandler.dispatch_event(locals(), 'after_epoch')

            if to_bool(Config.config['OUTPUTS']['save_model']) is True:
                ModelHandler.save_model(self.model, epoch)

        self._teardown()

    def _teardown(self):
        pass

    def setup(self):

        num_classes = 10

        # torch.manual_seed(1234)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        half_precision = to_bool(Config.get_option('ALGORITHM', 'half_precision', 'False'))
        max_img_channels = int(Config.get_option('ALGORITHM', 'max_img_channels', 1000))
        cholesky_weights = to_bool(Config.get_option('ARCHITECTURE', 'cholesky_weights', 'False'))

        train_images, test_images = Images.create_list_from_config()

        train_set = MnistDataset(
            self.device,
            images=train_images,
            regime='train',
            half_precision=half_precision,
            max_img_channels=max_img_channels
        )

        self.data_loaders['train'] = MnistDataLoader(
            train_set,
            shuffle=False,
            batch_size=int(Config.config['ALGORITHM']['train_batch_size']),
            pin_memory=True
        )

        test_set = MnistDataset(
            self.device,
            images=test_images,
            regime='test',
            half_precision=half_precision,
            max_img_channels=max_img_channels
        )

        self.data_loaders['test'] = MnistDataLoader(
            test_set,
            shuffle=False,
            batch_size=int(Config.config['ALGORITHM']['test_batch_size']),
            pin_memory=True
        )

        img_dims = train_set.tensor_size()

        arch_class_name = Config.config['ARCHITECTURE']['arch_class_name']
        model_class = class_for_name(arch_class_name)

        self.model = model_class(img_dims, num_classes, cholesky_weights=cholesky_weights)

        if to_bool(Config.config['OUTPUTS']['load_model']) is True:
            model_parameters_url = Config.config['OUTPUTS']['model_parameters_url']
            state_dict = ModelHandler.load_model(model_parameters_url)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)

        self.epochs = int(Config.config['ALGORITHM']['epochs'])

        target = 0

        precision = 1.e-5
        max_iter = 5e3
        ep = 1.e-0

        x0 = torch.randn(1, 6, 28, 28, 8, device=self.device)
        x = Variable(x0, requires_grad=True)

        save_interval = 500
        print_interval = 100

        diff = precision + 1
        i = 0

        for param in self.model.parameters():
            param.requires_grad = False

        saliency_path = os.path.join(Config.config['EXPERIMENT']['results_path'], 'saliency')

        if not os.path.isdir(saliency_path):
            os.makedirs(saliency_path)

        optimizer = Adam([x], lr=0.01, betas=(0.9, 0.999))

        reg_weight = 1.e-1

        while diff > precision and i < max_iter:

            optimizer.zero_grad()

            output = - (self.model.scores(x)[0][target] + reg_weight * torch.norm(x))

            output.backward()

            optimizer.step()

            # x = x + ep * x.grad
            # x = x.clone().detach().requires_grad_(True)

            if i == 0 or i % print_interval == 0:
                print(f'iteration {i:5d} -  output: {-output:1.3f}')
                # print(f'Target: {target}\t{i}/{max_iter} iterations updated')

            if i == 0 or i % save_interval == 0:

                saliency_map = np.squeeze(x.detach().numpy()).transpose((1, 2, 3, 0))
                save_nifti(os.path.join(saliency_path, f'saliency_{i}.nii.gz'), saliency_map, np.eye(4))

            i += 1

        if i == max_iter:
            print(f'Target: {target}\tReached max_iter')
        else:
            print(f'Target: {target}\tConverged in {i} iterations')



