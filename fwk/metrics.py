from util.lang import class_for_name
from util.path import absolute_path
import configparser


class Metric:

    def __init__(self) -> None:
        self.channels = list()

    def extract_metric(self, local_variables: dict) -> None:
        pass

    def print_metric(self) -> None:
        for channel in self.channels:
            self.print_for_channel(channel)

    def add_channel(self, channel):
        self.channels.append(channel)

    def print_for_channel(self, channel):
        pass

    @staticmethod
    def get_variable(local_variables: dict, key):

        if key not in local_variables.keys():
            raise RuntimeError(f'Metrics local variable {key} not found in scope.')

        variable = local_variables[key]

        return variable


class MetricsHandler:

    metrics = list()

    @classmethod
    def register_all_metrics(cls, metrics_url):
        config = configparser.ConfigParser(allow_no_value=True)
        # preserve CamelCase in keys
        # see https://docs.python.org/2/library/configparser.html#ConfigParser.RawConfigParser.optionxform
        config.optionxform = str
        config.read(absolute_path(metrics_url))

        for session in config.keys():

            for key in config[session]:
                metrics_class_name = key
                cls.add_metric(metrics_class_name)

    @classmethod
    def add_metric(cls, metric_class_name):

        metric = class_for_name(metric_class_name)()

        cls.metrics.append(metric)

    @classmethod
    def collect_metrics(cls, local_variables, event_type):

        for metric in cls.metrics:
            metric.extract_metric(local_variables, event_type)
            metric.print_metric(event_type)
