from fwk.config import Config
from fwk.metrics import MetricsHandler
from util.lang import class_for_name
from util.path import absolute_path
import os


class Experiment:

    @staticmethod
    def _results_url(config_url):
        results_url = os.path.dirname(config_url)
        os.makedirs(results_url, exist_ok=True)
        return results_url

    @ classmethod
    def run(cls, config_url):

        # create configuration object
        Config.set_config_from_url(absolute_path(config_url))
        Config.config['EXPERIMENT']['results_path'] = Experiment._results_url(config_url)

        # register the metrics handler
        MetricsHandler.register_metrics()

        # runs experiment
        experiment = class_for_name(Config.config['EXPERIMENT']['experiment_class_name'])
        experiment().execute()
