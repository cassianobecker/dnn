import os
from distutils.util import strtobool

import nibabel as nib
import numpy as np
import scipy.io as sio
import scipy.signal
import scipy.sparse

from sklearn.preprocessing import StandardScaler

from util.logging import get_logger, set_logger
from util.path import get_root

from dataset.hcp.downloaders import HcpDownloader, DtiDownloader
from dataset.hcp.converter import convert_surf_to_nifti


class HcpReader:

    def __init__(self, database_settings, params):

        log_furl = os.path.join(params['FILE']['experiment_path'], 'log', 'downloader.log')
        set_logger('HcpReader', database_settings['LOGGING']['dataloader_level'], log_furl)
        self.logger = get_logger('HcpReader')

        self.local_folder = database_settings['DIRECTORIES']['local_server_directory']
        self.delete_nii = strtobool(database_settings['DIRECTORIES']['delete_after_downloading'])
        self.hcp_downloader = HcpDownloader(database_settings)
        self.dti_downloader = DtiDownloader(database_settings)
        nib.imageglobals.logger = set_logger('Nibabel', database_settings['LOGGING']['nibabel_level'], log_furl)

        self.parcellation = params['PARCELLATION']['parcellation']
        self.inflation = params['SURFACE']['inflation']
        self.tr = float(params['FMRI']['tr'])
        self.fh = float(params['FMRI']['physio_sampling_rate'])
        self.physio_sampling_rate = int(params['FMRI']['physio_sampling_rate'])
        self.regress_physio = strtobool(params['FMRI']['regress_physio'])

    def load_subject_list(self, list_url):

        self.logger.info('loading subjects from ' + list_url)

        with open(list_url, 'r') as f:
            subjects = [s.strip() for s in f.readlines()]

        self.logger.info('loaded ' + str(len(subjects)) + ' subjects from: ' + list_url)

        return subjects

    def process_subject(self, subject, tasks):

        self.logger.info('processing subject {}'.format(subject))

        task_list = dict()
        for task in tasks:
            self.logger.debug('processing task {} ...'.format(task))
            task_dict = dict()

            task_dict['ts'] = self.get_fmri_time_series(subject, task)

            task_dict['cues'] = self.get_cue_array(subject, task, task_dict['ts'].shape[1])

            task_dict['vitals'] = self.get_vitals(subject, task)

            task_list[task] = task_dict

        data = dict()
        data['functional'] = task_list

        data['adjacency'] = self.get_adjacency(subject).tocsr()

        return data

    # ##################### VITALS ####################################

    def get_vitals(self, subject, task):

        vitals = dict()
        if self.regress_physio is True:
            try:

                self.logger.debug("reading vitals for " + subject)

                fname = 'tfMRI_' + task + '_Physio_log.txt'
                furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, fname)

                self.hcp_downloader.load(furl, subject)

                heart, resp = self.read_vitals(furl)
                vitals['heart'] = heart
                vitals['resp'] = resp

                self.logger.debug("done")

            except FileNotFoundError:
                msg = 'subject ' + subject + ' does not have physiological information'
                self.logger.warning(msg)
                raise SkipSubjectException(msg)

        return vitals

    def read_vitals(self, furl):

        try:
            with open(os.path.join(self.local_folder, furl)) as inp:
                phy = [line.strip().split('\t') for line in inp]
                resp = [int(phy_line[1]) for phy_line in phy]
                heart = [int(phy_line[2]) for phy_line in phy]
        except FileNotFoundError:
            msg = "file " + os.path.join(self.local_folder, furl) + " not found"
            self.logger.error(msg)
            raise SkipSubjectException(msg)

        return self.decimate(heart), self.decimate(resp)

    def decimate(self, signal):
        fl = 1 / self.tr
        fh = self.fh
        # decimation order
        n = 2
        q = int(fh / fl)
        signal_d = scipy.signal.decimate(np.array(signal[:-1]), q, n, zero_phase=True)
        return signal_d

    # ##################### TIME SERIES ####################################

    def get_fmri_time_series(self, subject, task):

        # get time series
        ts = self.load_raw_time_series(subject, task)

        # get parcellation
        parc_vector, parc_labels = self.get_parcellation(subject)

        # parcellate time series
        ts_p = self.parcellate(ts, parc_vector, parc_labels)

        # normalize
        ts_norm = self.normalize_time_series(ts_p)
        return ts_norm

    def load_raw_time_series(self, subject, task):

        self.logger.debug("loading time series for " + subject)

        fname = 'tfMRI_' + task + '_Atlas.dtseries.nii'
        furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, fname)

        self.hcp_downloader.load(furl, subject)

        try:
            furl = os.path.join(self.local_folder, furl)
            ts = np.array(nib.load(furl).get_data())
            if self.delete_nii:
                self.hcp_downloader.delete_dir(furl, subject)
        except:
            msg = "file " + furl + " not found."
            self.logger.error(msg)
            raise SkipSubjectException(msg)

        self.logger.debug("done")

        return ts

    def get_parcellation(self, subject):

        self.logger.debug("reading parcellation for " + subject)

        fpath = os.path.join('HCP_1200', subject, 'MNINonLinear', 'fsaverage_LR32k')

        suffixes = {'aparc': '.aparc.a2009s.32k_fs_LR.dlabel.nii',
                    'dense': '.aparc.a2009s.32k_fs_LR.dlabel.nii'}

        parc_furl = os.path.join(fpath, subject + suffixes[self.parcellation])

        self.hcp_downloader.load(parc_furl, subject)

        parc_vector, parc_labels = None, None

        try:
            parc_obj = nib.load(os.path.join(self.local_folder, parc_furl))
        except FileNotFoundError:
            msg = "File " + os.path.join(self.local_folder, parc_furl) + " not found"
            self.logger.error(msg)
            raise SkipSubjectException(msg)

        if self.parcellation == 'aparc':
            parc_vector = np.array(parc_obj.get_data()[0], dtype='int')
            table = parc_obj.header.matrix[0].named_maps.__next__().label_table
            parc_labels = [(region[1].key, region[1].label) for region in table.items()]

        if self.parcellation == 'dense':
            n_regions = len(parc_obj.get_data()[0])
            parc_vector = np.array(range(n_regions))
            parc_labels = [(i, i) for i in range(n_regions)]

        self.logger.debug("done")

        return parc_vector, parc_labels

    def parcellate(self, ts, parc_vector, parc_labels):

        self.logger.debug("performing parcellation")

        tst = ts[:, :parc_vector.shape[0]]

        parc_idx = list(np.unique(parc_vector))
        x_parc = None

        bad_regions = [label[0] for label in parc_labels if label[1] == '???']
        for bad_region in bad_regions:
            parc_idx.remove(bad_region)

        if self.parcellation == 'aparc':
            # build parcellated signal by taking the mean across voxels
            x_parc = np.array([np.mean(tst[:, (parc_vector == i).tolist()], axis=1) for i in parc_idx])

        if self.parcellation == 'dense':
            x_parc = tst.T

        self.logger.debug("done")

        return x_parc

    def normalize_time_series(self, ts_p):

        scaler = StandardScaler()
        ts_p = scaler.fit_transform(np.swapaxes(ts_p, 0, 1))
        return np.swapaxes(ts_p, 0, 1)

    # ##################### CUES ####################################

    def get_cue_array(self, subject, task, ts_length):

        self.logger.debug("reading cue signals for " + subject)

        cue_names = ['cue', 'lf', 'lh', 'rf', 'rh', 't']
        cue_events = dict()
        for cue_name in cue_names:
            furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'tfMRI_' + task, 'EVs',
                                cue_name + '.txt')
            self.hcp_downloader.load(furl, subject)
            cue_events[cue_name] = self.read_cue_events_file(os.path.join(self.local_folder, furl))

        cue_array = self.encode_cues(cue_events, ts_length)
        self.logger.debug("done")

        return cue_array[1:]

    def read_cue_events_file(self, furl):

        try:
            with open(furl) as inp:
                evs = [line.strip().split('\t') for line in inp]
                evs_t = [int(float(evi[0]) / self.tr) for evi in evs]
        except FileNotFoundError:
            msg = "events file " + os.path.join(self.local_folder, furl) + " not found"
            self.logger.error(msg)
            raise SkipSubjectException(msg)
        return evs_t

    def encode_cues(self, cue_events, ts_length):
        cue_array = np.zeros((len(cue_events), ts_length), dtype=int)
        for cue_idx, cue in enumerate(cue_events):
            events = cue_events[cue]
            for event_time in events:
                cue_array[cue_idx, event_time] = 1

        return cue_array

    # ##################### ADJACENCY MATRIX #########################################

    def get_adjacency(self, subject):

        adj = None

        if self.parcellation == 'aparc':
            adj = self.get_adjacency_dti(subject)

        if self.parcellation == 'dense':
            adj, coords = self.get_adjacency_mesh(subject)

        return adj

    def get_adjacency_mesh(self, subject):

        self.logger.debug("reading mesh adjacency matrix for" + subject)

        self.logger.debug("processing left hemisphere edges for " + subject)
        rows_L, cols_L, coords_L = self.get_adjacency_mesh_hemi('L', subject)
        self.logger.debug("done")

        self.logger.debug("processing right hemisphere edges for " + subject)
        rows_R, cols_R, coords_R = self.get_adjacency_mesh_hemi('R', subject)
        self.logger.debug("done")

        self.logger.debug("processing coordinates for " + subject)

        data = np.ones(len(rows_L) + len(rows_R))
        A = scipy.sparse.coo_matrix((data, (rows_L + rows_R, cols_L + cols_R)))

        new_coords = np.vstack((coords_L, coords_R))
        # new_coords = filter_surf_vertices(coords)

        self.logger.debug("done")

        return A, new_coords

    def get_adjacency_mesh_hemi(self, hemi, subject):

        fname = subject + '.' + hemi + '.' + self.inflation + '.32k_fs_LR.surf.gii'
        furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'fsaverage_LR32k', fname)
        self.hcp_downloader.load(furl, subject)

        try:
            img = nib.load(os.path.join(self.local_folder, furl))
        except FileNotFoundError:
            msg = "surface mesh file " + os.path.join(self.local_folder, furl) + " not found"
            self.logger.error(msg)
            raise SkipSubjectException(msg)

        coords = img.darrays[0].data
        faces = img.darrays[1].data.astype(int)

        rows, cols, new_coords = convert_surf_to_nifti(faces, coords, hemi)

        return rows, cols, new_coords

    def get_adjacency_dti(self, subject):

        self.logger.debug("reading dti adjacency matrix for " + subject)

        furl = os.path.join('HCP_1200', subject, 'MNINonLinear', 'Results', 'dMRI_CONN')
        file = os.path.join(furl, subject + '.aparc.a2009s.dti.conn.mat')
        token_url = os.path.join(furl, subject + '_404_token.txt')

        self.dti_downloader.load(file, token_url, subject)

        # looks for local subject-specific DTI matrix
        try:
            dti_matrix = sio.loadmat(os.path.join(self.local_folder, file)).get('S')

        except FileNotFoundError:

            # if not found, loads average DTI matrix from local resources directory
            msg = "specific DTI matrix for " + subject + "," + self.parcellation + " not available, using average."
            self.logger.info(msg)

            relative_avg_furl = os.path.join('dataset', 'hcp', 'res', 'average1.aparc.a2009s.dti.conn.mat')
            avg_furl = os.path.join(get_root(), relative_avg_furl)

            try:
                dti_matrix = sio.loadmat(avg_furl).get('S')

            except:
                msg = self.logger.error("average DTI file " + avg_furl + " not found")
                self.logger.error(msg)
                raise SkipSubjectException(msg)

        dti_coo = scipy.sparse.coo_matrix(dti_matrix)

        self.logger.debug("done")

        return dti_coo


class SkipSubjectException(Exception):
    def __init(self, msg):
        super(SkipSubjectException, self).__init__(msg)
