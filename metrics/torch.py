import psutil
from fwk.metrics import Metric


def print_cpu_memory():

    divider = 1024. * 1024.
    percent = psutil.virtual_memory().percent
    available = psutil.virtual_memory().available / divider
    used = psutil.virtual_memory().used / divider

    mem_str = 'memory usage: {:.2f}%, used: {:.1f} MB, available: {:.1f} MB' \
        .format(percent, used, available)

    return mem_str


class GradientMetrics(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.gradient_norm = None
        self.batch_idx = None
        self.epoch = None

    def extract_metric(self, local_variables: dict):

        model = self.get_variable(local_variables, 'model')
        self.batch_idx = self.get_variable(local_variables, 'batch_idx')
        self.epoch = self.get_variable(local_variables, 'epoch')
        self.gradient_norm = self._compute_norm(model)

    @staticmethod
    def _compute_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.norm(2).item() if p.grad is not None else 0
            total_norm += param_norm ** 2
        total_norm = total_norm ** (1. / 2)

        return total_norm

    def print_metric(self):
        print(f'Epoch: {self.epoch} batch: {self.batch_idx} gradient norm: {self.gradient_norm}')



