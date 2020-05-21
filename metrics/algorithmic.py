import math
from fwk.metrics import Metric
from fwk.config import Config


class EpochCounter(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.epoch = None
        self.total_epochs = None

    def on_before_epoch(self, local_variables):
        self.epoch = local_variables['epoch']
        self.total_epochs = int(Config.config['ALGORITHM']['epochs'])
        self.print_metric()

    def text_record(self):
        record_str = f'\n---- epoch {self.epoch + 1:03d} of {self.total_epochs:03d} --------------------\n'
        return record_str


class TrainBatchCounter(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.number_of_subjects = None
        self.batch_idx = None
        self.subjects_per_batch = None
        self.number_of_batches = None
        self.regime = 'train'

    def on_before_train_batch(self, local_variables):
        self.number_of_subjects = len(local_variables['self'].data_loaders[self.regime].dataset.subjects)
        self.subjects_per_batch = int(Config.config['ALGORITHM'][f'{self.regime}_batch_size'])
        self.batch_idx = local_variables['batch_idx']
        self.number_of_batches = math.ceil(self.number_of_subjects / self.subjects_per_batch)
        self.print_metric()

    def text_record(self):
        text_record = f'\n  batch {self.batch_idx + 1:03d} of {self.number_of_batches:03d} ({self.regime})\n'
        return text_record


class TestBatchCounter(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.number_of_subjects = None
        self.batch_idx = None
        self.subjects_per_batch = None
        self.number_of_batches = None
        self.regime = 'test'

    def on_before_test_batch(self, local_variables):
        self.number_of_subjects = len(local_variables['self'].data_loaders[self.regime].dataset.subjects)
        self.subjects_per_batch = int(Config.config['ALGORITHM'][f'{self.regime}_batch_size'])
        self.batch_idx = local_variables['batch_idx']
        self.number_of_batches = math.ceil(self.number_of_subjects / self.subjects_per_batch)
        self.print_metric()

    def text_record(self):
        text_record = f'\n  batch {self.batch_idx + 1:03d} of {self.number_of_batches:03d} ({self.regime})\n'
        return text_record


class ImageTrainBatchCounter(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.number_of_images = None
        self.batch_idx = None
        self.images_per_batch = None
        self.number_of_batches = None
        self.regime = 'train'

    def on_before_train_batch(self, local_variables):
        self.number_of_images = len(local_variables['self'].data_loaders[self.regime].dataset.images)
        self.images_per_batch = int(Config.config['ALGORITHM'][f'{self.regime}_batch_size'])
        self.batch_idx = local_variables['batch_idx']
        self.number_of_batches = int(self.number_of_images / self.images_per_batch)
        self.print_metric()

    def text_record(self):
        text_record = f'\n  batch {self.batch_idx + 1:03d} of {self.number_of_batches:03d} ({self.regime})\n'
        return text_record


class ImageTestBatchCounter(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.number_of_images = None
        self.batch_idx = None
        self.images_per_batch = None
        self.number_of_batches = None
        self.regime = 'test'

    def on_before_test_batch(self, local_variables):
        self.number_of_images = len(local_variables['self'].data_loaders[self.regime].dataset.images)
        self.images_per_batch = int(Config.config['ALGORITHM'][f'{self.regime}_batch_size'])
        self.batch_idx = local_variables['batch_idx']
        self.number_of_batches = int(self.number_of_images / self.images_per_batch)
        self.print_metric()

    def text_record(self):
        text_record = f'\n  batch {self.batch_idx + 1:03d} of {self.number_of_batches:03d} ({self.regime})\n'
        return text_record


class BatchLoss(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.loss = None

    def on_after_train_batch(self, local_variables):
        self.loss = local_variables['loss'].item()
        self.print_metric()

    def text_record(self):
        text_record = f'    batch loss: {self.loss:.3e}\n'
        return text_record


class EpochLoss(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.loss = None

    def on_after_train_batch(self, local_variables):
        self.loss += local_variables['loss'].item()

    def on_before_epoch(self, local_variables):
        self.loss = 0

    def on_after_epoch(self, local_variables):
        self.print_metric()

    def text_record(self):
        text_record = f'\n    epoch loss: {self.loss:.3e}\n'
        return text_record

    def numpy_record(self, records=None):

        if 'epoch_loss' not in records.keys():
            records['epoch_loss'] = list()

        records['epoch_loss'].append(self.loss)

        return records


class GradientMetrics(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.gradient_norm = None
        self.batch_idx = None
        self.epoch = None

    def on_after_train_batch(self, local_variables):
        model = local_variables['self'].model
        self.batch_idx = local_variables['batch_idx']
        self.epoch = local_variables['epoch']
        self.gradient_norm = self._compute_norm(model)
        self.print_metric()

    def text_record(self):
        text_record = f'    gradient norm: {self.gradient_norm:.3e}\n'
        return text_record

    def numpy_record(self, records=None):

        if 'gradient_norm' not in records.keys():
            records['gradient_norm'] = []

        if self.batch_idx == 0:
            records['gradient_norm'].append([self.gradient_norm])
        else:
            records['gradient_norm'][self.epoch].append(self.gradient_norm)

        return records

    @staticmethod
    def _compute_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.norm(2).item() if p.grad is not None else 0
            total_norm += param_norm ** 2
        total_norm = total_norm ** (1. / 2)

        return total_norm


class NumberOfParameters(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.model = None

    def on_after_setup(self, local_variables):
        self.model = local_variables['self'].model
        self.print_metric()

    def text_record(self):

        total_str = f'number of parameters {self._total_parameters(self.model):1.3e}\n'
        total_trainable_str = f'number of trainable parameters {self._total_trainable_parameters(self.model):1.3e}\n'

        return total_str + total_trainable_str

    @staticmethod
    def _total_parameters(model):
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def _total_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
