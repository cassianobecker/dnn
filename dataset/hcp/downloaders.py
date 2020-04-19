import os
import boto3

from util.logging import get_logger, set_logger
from fwk.config import Config


class HcpDiffusionDownloader:

    def __init__(self):

        log_furl = get_logger('HcpReader').handlers[0].stream.name
        set_logger('DiffusionDownloader', Config.config['LOGGING']['downloader_level'], log_furl)
        self.logger = get_logger('DiffusionDownloader')

    def load(self, path, subject):

        local_path = os.path.join(os.path.expanduser(Config.config['DATABASE']['local_server_directory']), path)
        if os.path.isfile(local_path):
            self.logger.debug("file found in: " + local_path)
        else:
            bucket = 'hcp-openaccess'
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                boto3.resource('s3').Bucket(bucket).download_file(path, local_path)
            except Exception as e:
                print(e)

    def delete_dir(self, path):
        path = os.path.join(os.path.expanduser(Config.config['DATABASE']['local_server_directory']), path)
        if os.path.isfile(path):
            os.remove(path)
