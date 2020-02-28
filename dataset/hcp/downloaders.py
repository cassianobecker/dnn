import os
import requests

from util.logging import get_logger, set_logger


class HcpDownloader:
    """
    Enables downloading patient files from the HCP database through HTTP get requests if they are unavailable locally.
    :param settings: ConfigParser that contains server, directory and credential info and logging levels
    :param test: boolean to differentiate the loggers for training and test sets
    """

    def __init__(self, database_settings):

        self.settings = database_settings

        log_furl = get_logger('HcpReader').handlers[0].stream.name
        set_logger('HcpDownloader', database_settings['LOGGING']['downloader_level'], log_furl)
        self.logger = get_logger('HcpDownloader')

    def load(self, path, subject):
        """
        Checks for file in the specified path. If file is unavailable, downloads it from the HCP database.
        :param path: local path to check for file and download to if unavailable
        :return: None
        """
        path = os.path.join(self.settings['DIRECTORIES']['local_server_directory'], path)
        if os.path.isfile(path):

            self.logger.debug("file found in: " + path)

        else:

            self.logger.info("file not found in: " + path)

            # subject = path.split('/')[5]
            key = path.split('/MNINonLinear/')[1]
            url = self.settings['SERVERS']['hcp_server_url'].format(subject, subject, subject) + key

            self.logger.info("remote download from server: " + url)

            if self.settings['CREDENTIALS']['hcp_server_username'] == '' \
                    or self.settings['CREDENTIALS']['hcp_server_password'] == '':

                self.logger.error("HCP server credentials are empty")

            r = requests.get(url,
                             auth=(self.settings['CREDENTIALS']['hcp_server_username'],
                                   self.settings['CREDENTIALS']['hcp_server_password']),
                             stream=True)

            if r.status_code == 200:

                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    self.logger.debug("writing to path: " + path)
                    f.write(r.content)
                    self.logger.debug("writing to " + path + 'completed')

            else:
                self.logger.error("request unsuccessful: Error " + str(r.status_code))

    def delete_dir(self, path):
        path = os.path.join(self.settings['DIRECTORIES']['local_server_directory'], path)
        if os.path.isfile(path):
            os.remove(path)


class DtiDownloader:
    """
    Enables downloading DTI files from a web repository through HTTP get requests if they are unavailable locally.
    :param settings: configparser that contains server, directory and credential info and logging levels
    :param test: Boolean to differentiate the loggers for training and test sets
    """

    def __init__(self, database_settings):

        self.base_path = database_settings['SERVERS']['dti_server_url']
        self.local_path = database_settings['DIRECTORIES']['local_server_directory']

        log_furl = get_logger('HcpReader').handlers[0].stream.name
        set_logger('DtiDownloader', database_settings['LOGGING']['downloader_level'], log_furl)
        self.logger = get_logger('DtiDownloader')


    def load(self, path, token_url):
        """
        Checks for file in the specified path. If file is unavailable, downloads it from the HCP database.
        :param path: local path to check for file and download to if unavailable
        :param token_url: local path to check for token and download to if unavailable
        :return: None
        """
        path = os.path.join(self.local_path, path)
        token_url = os.path.join(self.local_path, token_url)

        # looks for specific subject file
        if os.path.isfile(path):
            self.logger.debug("file found in: " + path)

        # if not, looks for token to prevent download retrials
        elif os.path.isfile(token_url):
            self.logger.debug("token found in: " + token_url)

        # if no specific file nor token, attempts to download specific file
        else:

            self.logger.info("file not found in: " + path)
            subject = path.split('/')[5]

            key = path.split('/MNINonLinear/')[1]
            temp = key.split('/', 1)
            key = temp[0] + '/' + subject + '/' + temp[1]
            url = self.base_path + key

            r = requests.get(url)

            self.logger.info("remote download from server: " + url)

            # if download successful
            if r.status_code == 200:

                # save subject specific file
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    self.logger.debug("writing to path: " + path)
                    f.write(r.content)
                    self.logger.debug("writing to " + path + 'completed')

            # if download unsuccessful
            else:
                self.logger.warning("DTI request unsuccessful: Error " + str(r.status_code))

                # creates token for future requests
                self.logger.debug("creating token in path: " + token_url)
                dmri_dir = os.path.join(self.local_path, 'HCP_1200', subject, 'MNINonLinear', 'Results', 'dMRI_CONN')
                if not os.path.exists(dmri_dir):
                    os.makedirs(dmri_dir)
                open(token_url, 'a').close()
                self.logger.debug("creating token in " + token_url + 'completed')
