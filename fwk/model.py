import torch
import os
from fwk.config import Config


class ModelHandler:

    @classmethod
    def load_model(cls, model_parameters_url, epoch=None):
        return torch.load(os.path.expanduser(model_parameters_url), map_location=torch.device('cpu'))

    @classmethod
    def save_model(cls, model, epoch):
        # model_url = cls._make_path_for_model(epoch)
        # torch.save(model.state_dict(), model_url)
        model_url_latest = cls._make_path_for_model(epoch=None)
        torch.save(model.state_dict(), model_url_latest)

    @classmethod
    def _check_path_for_model(cls, epoch):
        model_path, model_name = cls._url_components_for_model(epoch)
        model_url = os.path.join(model_path, model_name)
        if not os.path.exists(model_url):
            raise FileNotFoundError(f'Model file not found: {model_url}')

        return model_url

    @classmethod
    def _make_path_for_model(cls, epoch):
        model_path, model_name = cls._url_components_for_model(epoch)
        os.makedirs(model_path, exist_ok=True)
        return os.path.join(model_path, model_name)

    @classmethod
    def _url_components_for_model(cls, epoch):

        model_name = 'model.pt' if epoch is None else f'model_{epoch}.pt'
        results_path = Config.config['EXPERIMENT']['results_path']
        model_path = os.path.join(results_path, 'model')

        return model_path, model_name
