import psutil
from fwk.metrics import Metric
import torch


def cpu_memory():

    divider = 1024. * 1024.
    percent = psutil.virtual_memory().percent
    available = psutil.virtual_memory().available / divider
    used = psutil.virtual_memory().used / divider

    return percent, available, used


class GpuMemory(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.variable = None
        self.total = None
        self.cached = None
        self.allocated = None
        self.free_inside_cache = None
        self.device = None

    def extract_metric(self, local_variables: dict, event_type):

        if event_type == 'AFTER_TRAIN_BATCH' or event_type == 'AFTER_TEST_BATCH':

            self.device = local_variables['self'].device.type

            if self.device == 'gpu':

                self.total = torch.cuda.get_device_properties(0).total_memory
                self.cached = torch.cuda.memory_cached(0)
                self.allocated = torch.cuda.memory_allocated(0)
                self.free_inside_cache = self.cached - self.allocated

    def print_metric(self, event_type):

        if event_type == 'AFTER_TRAIN_BATCH' or event_type == 'AFTER_TEST_BATCH':

            if self.device == 'gpu':

                gpu_str = f'    GPU Memory: total {self.total}, allocated {self.allocated}, cached {self.cached}'
                print(gpu_str)


class CpuMemory(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.percent = None
        self.available = None
        self.used = None

    def extract_metric(self, local_variables: dict, event_type):

        if event_type == 'AFTER_TRAIN_BATCH' or event_type == 'AFTER_TEST_BATCH':
            self.percent, self. available, self.used = cpu_memory()

    def print_metric(self, event_type):

        if event_type == 'AFTER_TRAIN_BATCH' or event_type == 'AFTER_TEST_BATCH':
            mem_str = '    memory usage: {:.2f}%, used: {:.1f} MB, available: {:.1f} MB' \
                .format(self.percent, self.used, self.available)

            print(mem_str)
