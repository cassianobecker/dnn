import os
from fwk.config import Config


class Covariates:

    def __init__(self):
        self.processing_folder = os.path.expanduser(Config.config['DATABASE']['local_processing_directory'])

    def _covariate_url(self, field, subject):
        return os.path.join(self.processing_folder, subject, 'tracts', field + '.txt')

    def _read_float(self, field, subject):
        with open(self._covariate_url(field, subject), 'r') as f:
            lines = (line.strip() for line in f if line)
            value = [float(line.strip()) for line in lines]
            return value[0]

    def value(self, field, subject, regression=False):
        value = self._read_float(field, subject)
        return value
