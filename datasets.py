from pathlib import Path
import platform

import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import mne
from os import listdir
from os.path import isfile, join, exists
import json
import re
from tqdm import tqdm
import numpy as np
from utils.file_patient_session import filename_to_patient_id, filename_to_session
import numpy as np
import h5py
import random
from settings import *

mne.set_log_level("WARNING")

# todo read this file from some common place, e.g. Preprocess class


class WindowedEEGDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        super(WindowedEEGDataset, self).__init__()
        # self.labels = pd.read_csv("personality_57.csv")[['hash', 'NEO N', 'NEO E', 'NEO O', 'NEO U', 'NEO S']]
        self.labels = pd.read_csv("personality_57_rec.csv")[
            ['hash', 'REC N', 'REC E', 'REC O', 'REC U', 'REC S']]
        for column in self.labels.columns:
            if column == 'hash':
                continue
        self.data_dir = data_dir
        self.transform = transform
        self.hf_name = f"windowed_EEG_dataset_{segment_length}_{sfreq}_{overlap}.hdf5"
        self.create_dataset_file()
        with h5py.File(join(self.data_dir, self.hf_name), 'r') as f:
            self.len = f['data'].shape[0]
        self.data = None

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.data is None:
            self.data = h5py.File(join(self.data_dir, self.hf_name), 'r')

        data = self.data['data'][idx]
        label = self.data['label'][idx]
        if self.transform:
            data = self.transform(data)
        sample = (torch.FloatTensor(data), torch.FloatTensor(label))

        return sample

    def train_val_split(self, val_split=0.25):
        with h5py.File(join(self.data_dir, self.hf_name), 'r') as f:
            hashes_all = f['hash']
            hashes = list(set(hashes_all))
            random.shuffle(hashes)
            train_hashes = hashes[int(len(hashes)*val_split):]
            val_hashes = hashes[:int(len(hashes)*val_split)]
            train_idx = []
            val_idx = []
            for hash in train_hashes:
                indices = [i for i, x in enumerate(hashes_all) if x == hash]
                train_idx += indices
            for hash in val_hashes:
                indices = [i for i, x in enumerate(hashes_all) if x == hash]
                val_idx += indices
            random.shuffle(train_idx)
            random.shuffle(val_idx)
            return train_idx, val_idx

    def create_dataset_file(self):
        if isfile(join(self.data_dir, self.hf_name)):
            return
        with h5py.File(join(self.data_dir, self.hf_name), 'w') as hf:
            files = [f for f in listdir(path_seq) if isfile(
                join(path_seq, f)) and f.lower().endswith(('.edf'))]

            for file in tqdm(files):
                raw = mne.io.read_raw_edf(path_seq / file)
                segmented = mne.make_fixed_length_epochs(raw=raw, duration=epoch_len, preload=True,
                                                         overlap=epoch_len * overlap)
                data = segmented.get_data()
                hash = filename_to_patient_id(file)
                big_5_params = self.labels.loc[self.labels['hash'] == hash].drop([
                    'hash'], axis=1)
                big_5_params = np.squeeze(big_5_params.to_numpy())
                label = np.vstack([big_5_params] * data.shape[0])
                # data = torch.unsqueeze(data, 0)
                # label = torch.unsqueeze(label, 0)
                data_shape = (None,) + (*data.shape,)[1:]
                label_shape = (None,) + (*label.shape,)[1:]
                if np.isnan(np.sum(label)):
                    continue
                if not hf.keys():
                    compression_opts = 4
                    hf.create_dataset('data', data=data,
                                      compression="gzip", compression_opts=compression_opts, maxshape=data_shape)
                    hf.create_dataset('label', data=label,
                                      compression="gzip", compression_opts=compression_opts, maxshape=label_shape)
                    hf.create_dataset('hash', data=[hash],
                                      compression="gzip", compression_opts=compression_opts, maxshape=(None,))
                else:
                    hf['data'].resize(
                        (hf['data'].shape[0] + data.shape[0]), axis=0)
                    hf['data'][-data.shape[0]:] = data

                    hf['label'].resize(
                        (hf['label'].shape[0] + label.shape[0]), axis=0)
                    hf['label'][-label.shape[0]:] = label
                    hf['hash'].resize(
                        (hf['hash'].shape[0] + 1), axis=0)
                    hf['hash'][-1:] = hash


class WindowedSequenceEEGDataset(Dataset):
    # TODO synthesize dataset by segmenting data again to increase number of datapoints

    def __init__(self, index_file, root_dir, transform=None):
        with open(index_file, 'r') as f:
            self.index = json.load(f)
        # self.labels = pd.read_csv("personality_57.csv")[['hash', 'NEO N', 'NEO E', 'NEO O', 'NEO U', 'NEO S']]
        self.labels = pd.read_csv("personality_57_rec.csv")[
            ['hash', 'REC N', 'REC E', 'REC O', 'REC U', 'REC S']]
        for column in self.labels.columns:
            if column == 'hash':
                continue
            # self.labels[column] = (self.labels[column] - self.labels[column].mean()) / self.labels[column].std()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        info = self.index[idx]
        hash = info["hash"]
        session = int(info["session"])

        raw = mne.io.read_raw_edf(
            join(self.root_dir, f"{hash}_rest_standarized_T{session:02d}.edf"))
        segmented = mne.make_fixed_length_epochs(raw=raw, duration=epoch_len, preload=True,
                                                 overlap=epoch_len * overlap)
        data = segmented.get_data()

        # big_5_params = self.labels.loc[self.labels['hash'] == hash][['NEO N', 'NEO E', 'NEO O', 'NEO U', 'NEO S']]
        big_5_params = self.labels.loc[self.labels['hash'] == hash][[
            'REC N', 'REC E', 'REC O', 'REC U', 'REC S']]
        big_5_params = big_5_params.to_numpy()
        sb5 = big_5_params.shape
        sample = [torch.FloatTensor(data),
                  torch.FloatTensor(big_5_params)]  # {'data': data, 'big_5_params': big_5_params}

        if self.transform:
            sample = self.transform(sample)

        return sample


# def create_windowed_index():
#     # index_path = join(path, 'windowed_index.json')
#     index_path = path / 'windowed_index.json'
#     if not exists(index_path):
#         files = [f for f in listdir(path) if isfile(
#             join(path, f)) and f.lower().endswith(('.edf'))]
#         index = []
#         for file in tqdm(files):
#             raw = mne.io.read_raw_edf(path / file)
#             segmented = mne.make_fixed_length_epochs(raw=raw, duration=epoch_len, preload=True,
#                                                      overlap=epoch_len * overlap)
#             for s in segmented.selection:
#                 index.append(
#                     {"hash": filename_to_patient_id(file), "session": filename_to_session(file), "selection": str(s)})
#         json_object = json.dumps(index, indent=4)
#         with open(index_path, "w") as outfile:
#             outfile.write(json_object)


def create_seq_windowed_index():
    index_path = path / 'seq_windowed_index.json'
    if not exists(index_path):
        files = [f for f in listdir(path) if isfile(
            join(path, f)) and f.lower().endswith(('.edf'))]
        index = []
        for file in tqdm(files):
            raw = mne.io.read_raw_edf(join(path, file))
            segmented = mne.make_fixed_length_epochs(raw=raw, duration=epoch_len, preload=True,
                                                     overlap=epoch_len * overlap)
            index.append({"hash": filename_to_patient_id(file),
                         "session": filename_to_session(file)})
        json_object = json.dumps(index, indent=4)
        with open(index_path, "w") as outfile:
            outfile.write(json_object)


if __name__ == "__main__":
    # # todo zmienne do klas i jako parametry!!!!
    # if platform.node().startswith('LAPTOP-0TK'):
    #     path = Path().absolute() / 'data'
    # elif platform.node().startswith('Igors-MacBook-Pro') or platform.node().startswith('igor-podolak-6.laptop.matinf'):
    #     DATA_ROOT_DIR = Path('/Users/igor/data')
    #     # info nazwa kartoteki z plikami -- wartosc w parametrze wywolania --datadir
    #     #     datadir = f"{DATA_ROOT_DIR}/personality_traits/RESTS_gr87"
    #     # datadir = DATA_ROOT_DIR / 'personality_traits' / 'RESTS_gr87'
    #     standarized_dir = DATA_ROOT_DIR / 'personality_traits' / 'RESTS_gr87_standarized'
    #     path = DATA_ROOT_DIR

    # create_windowed_index()
    wed = WindowedEEGDataset(path)
    wed.create_dataset_file()
    # print(wed.__getitem__(42))
    # create_seq_windowed_index()
    # seq_wed = WindowedSequenceEEGDataset(path)
