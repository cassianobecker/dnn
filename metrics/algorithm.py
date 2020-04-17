from fwk.metrics import Metric
from fwk.config import Config


class EpochCounter(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.epoch = None
        self.total_epochs =  None

    def extract_metric(self, local_variables: dict, event_type):

        if event_type == 'BEFORE_EPOCH':
            self.epoch = local_variables['epoch']
            self.total_epochs = Config.config['ALGORITHM']['epochs']

    def print_metric(self, event_type):

        if event_type == 'BEFORE_EPOCH':
            print('')
            print(f'-- epoch {self.epoch + 1} of {self.total_epochs} -------------------------------')


class Loss(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.loss = None

    def extract_metric(self, local_variables: dict, event_type):

        if event_type == 'AFTER_TRAIN_BATCH':
            self.loss += local_variables['loss'].item()

        if event_type == 'BEFORE_EPOCH':
            self.loss = 0

    def print_metric(self, event_type):

        if event_type == 'AFTER EPOCH':
            print(f'Loss: {self.loss}')


class GradientMetrics(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.gradient_norm = None
        self.batch_idx = None
        self.epoch = None

    def extract_metric(self, local_variables: dict, event_type):

        if event_type == 'AFTER_TRAIN_BATCH':
            model = local_variables['self'].model
            self.batch_idx = local_variables['batch_idx']
            self.epoch = local_variables['epoch']
            self.gradient_norm = self._compute_norm(model)

    @staticmethod
    def _compute_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.norm(2).item() if p.grad is not None else 0
            total_norm += param_norm ** 2
        total_norm = total_norm ** (1. / 2)

        return total_norm

    def print_metric(self, event_type):

        if event_type == 'AFTER_TRAIN_BATCH':
            print(f'    gradient norm: {self.gradient_norm:.3e}')


class NumberOfParameters(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.model = None

    def extract_metric(self, local_variables: dict, event_type):

        if event_type == 'AFTER_SETUP':
            self.model = local_variables['self'].model

    def print_metric(self, event_type):

        if event_type == 'AFTER_SETUP':

            total_str = f'Total parameters {self._total_parameters(self.model):1.3e}'
            total_trainable_str = f'Total trainable parameters {self._total_trainable_parameters(self.model):1.3e}'
            print(total_str)
            print(total_trainable_str)

    def _total_parameters(self, model):
        return sum(p.numel() for p in model.parameters())

    def _total_trainable_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
