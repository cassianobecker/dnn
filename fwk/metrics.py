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

    metrics = dict()

    @classmethod
    def register_all_metrics(cls, metrics_url):
        config = configparser.ConfigParser(allow_no_value=True)
        # preserve CamelCase in keys
        # see https://docs.python.org/2/library/configparser.html#ConfigParser.RawConfigParser.optionxform
        config.optionxform = str
        config.read(absolute_path(metrics_url))

        for session in config.keys():
            event_type = session

            for key in config[session]:

                metrics_class_name = key
                cls.add_metric(metrics_class_name, event_type)

    @classmethod
    def add_metric(cls, metric_class_name, event_type):

        metric = class_for_name(metric_class_name)()

        if event_type not in cls.metrics.keys():
            cls.metrics[event_type] = list()

        cls.metrics[event_type].append(metric)

    @classmethod
    def collect_metrics(cls, local_variables, event_type):

        if event_type in cls.metrics.keys():
            for metric in cls.metrics[event_type]:
                metric.extract_metric(local_variables)
                metric.print_metric()

