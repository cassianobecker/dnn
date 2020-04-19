import psutil
from fwk.metrics import Metric
import torch


class GpuMemory(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.variable = None
        self.total = None
        self.cached = None
        self.allocated = None
        self.free_inside_cache = None
        self.device = None

    def on_after_train_batch(self, local_variables):
        self._on_after_batch(local_variables)

    def on_after_test_batch(self, local_variables):
        self._on_after_batch(local_variables)

    def _on_after_batch(self, local_variables):
        self.device = local_variables['self'].device.type

        if self.device == 'gpu':
            self.total = torch.cuda.get_device_properties(0).total_memory
            self.cached = torch.cuda.memory_cached(0)
            self.allocated = torch.cuda.memory_allocated(0)
            self.free_inside_cache = self.cached - self.allocated

        self.print_metric()

    def text_record(self):

        if self.device == 'gpu':
            gpu_str = f'    gpu memory: total {self.total}, allocated {self.allocated}, cached {self.cached}\n'
        else:
            gpu_str = ''

        return gpu_str


class CpuMemory(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.percent = None
        self.available = None
        self.used = None

    def on_after_train_batch(self, local_variables):
        self._on_after_batch(local_variables)

    def on_after_test_batch(self, local_variables):
        self._on_after_batch(local_variables)

    def _on_after_batch(self, _):
        self.percent, self. available, self.used = self.cpu_memory()
        self.print_metric()

    def text_record(self):
        mem_str = '    cpu memory usage: {:.2f}%, used: {:.1f} MB, available: {:.1f} MB\n'\
            .format(self.percent, self.used, self.available)
        return mem_str

    @staticmethod
    def cpu_memory():
        divider = 1024. * 1024.
        percent = psutil.virtual_memory().percent
        available = psutil.virtual_memory().available / divider
        used = psutil.virtual_memory().used / divider

        return percent, available, used
