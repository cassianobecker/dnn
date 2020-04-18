from fwk.config import Config
import os
import numpy as np
from util.lang import is_method_implemented


class Renderer:

    def print_record(self, record_str):
        pass


class TextFileRenderer(Renderer):

    def __init__(self, observer_name) -> None:
        super().__init__()

        renderer_url = Config.config['EXPERIMENT']['results_path']

        self.url = os.path.join(renderer_url, observer_name + '.txt')

    def print_record(self, metric):

        record_str = metric.text_record()

        file = open(self.url, "a")
        file.write(record_str)
        file.close()


class NumpyRenderer(Renderer):

    def __init__(self, observer_name) -> None:
        super().__init__()
        renderer_url = Config.config['EXPERIMENT']['results_path']
        self.url = os.path.join(renderer_url, observer_name)
        self.records = dict()

    def print_record(self, metric):

        if is_method_implemented(metric, 'numpy_record'):
            self.records = metric.numpy_record(self.records)

        arrays = dict()

        for key in self.records.keys():
            arrays[key] = np.array(self.records[key])

        np.savez_compressed(self.url, **arrays)
