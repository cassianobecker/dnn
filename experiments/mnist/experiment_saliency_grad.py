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

        for param in self.model.parameters():
            param.requires_grad = False

        salience_path = os.path.join(Config.config['EXPERIMENT']['results_path'], 'saliency')
        if not os.path.isdir(salience_path):
            os.makedirs(salience_path)

        self.model.eval()

        xs, ys, ss = next(iter(self.data_loaders['test']))
        xs = xs.to(self.device).type(torch.float32)
        # for i in range(xs.shape[0]):  # can only run .backward() once on tensor
        #     x = xs.detach()
        #     x.requires_grad = True
        #     score = self.model(x)
        #     score_max_index = score[i].argmax()
        #     score_max = score[i, score_max_index]
        #     score_max.backward()
        #     salience_map = x.grad[i].abs().detach().numpy().transpose((1, 2, 3, 0))
        #     save_nifti(os.path.join(salience_path, f'saliency_{i}.nii.gz'), salience_map, np.eye(4))
        for i in range(xs.shape[0]):  # can only run .backward() once on tensor
            x = xs[i].detach().unsqueeze(0)
            x.requires_grad = True
            score = self.model(x)
            score_max_index = score[0].argmax()
            score_max = score[0, score_max_index]
            score_max.backward()
            salience_map = x.grad[0].abs().detach().numpy().transpose((1, 2, 3, 0)) * 1000000
            save_nifti(os.path.join(salience_path, f'saliency_{i}.nii.gz'), salience_map, np.eye(4))
