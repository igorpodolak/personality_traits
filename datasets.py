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

mne.set_log_level("WARNING")

"""
def filename_to_patient_id(filename):
    pattern = r'(\w{8})_rest_'
    match = re.search(pattern, filename)
    return match.group(1)


def filename_to_session(filename):
    # todo repair filename_to_patient_id() and filename_to_session to work irrespective
    #  to directory and move both to some utilities file
    pattern = r'_rest_standardized_T(\d\d?)'
    match = re.search(pattern, filename)
    return match.group(1)
"""

# todo ŻADNYCH ZMIENNYCH GLOBALNYCH!
Df = pd.read_csv("personality_57.csv")
segment_length = 2048
sfreq = 500
epoch_len = segment_length / sfreq
overlap = 0.5


class WindowedEEGDataset(Dataset):

    def __init__(self, index_file, root_dir, transform=None):
        with open(index_file, 'r') as f:
            self.index = json.load(f)
        self.labels = pd.read_csv("personality_57.csv")[['hash', 'NEO N', 'NEO E', 'NEO O', 'NEO U', 'NEO S']]
        for column in self.labels.columns:
            if column == 'hash':
                continue
            self.labels[column] = (self.labels[column] - self.labels[column].mean()) / self.labels[column].std()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        info = self.index[idx]
        hash = info["hash"]
        session = info["session"]
        selection = int(info["selection"])

        file_path = self.root_dir / f"{hash}_rest_standardized_T{session}.edf"
        raw = mne.io.read_raw_edf(file_path)
        segmented = mne.make_fixed_length_epochs(raw=raw, duration=epoch_len, preload=True,
                                                 overlap=epoch_len * overlap)
        data = segmented.get_data()[selection]
        big_5_params = self.labels.loc[self.labels['hash'] == hash].drop(['hash'], axis=1)
        big_5_params = np.squeeze(big_5_params.to_numpy())

        # sample = [torch.FloatTensor(np.expand_dims(data, axis=0)), torch.FloatTensor(big_5_params)] # {'data': data, 'big_5_params': big_5_params}
        sample = [torch.FloatTensor(data), torch.FloatTensor(big_5_params)]
        # if sample[0].size()[1] != 2048:
        #     print(f"{file_path}::{selection} has incorrect length of {sample[0].size()[1]}")
        #     exit()
        if self.transform:
            sample = self.transform(sample)

        return sample


class WindowedSequenceEEGDataset(Dataset):
    # TODO synthesize dataset by segmenting data again to increase number of datapoints

    def __init__(self, index_file, root_dir, transform=None):
        with open(index_file, 'r') as f:
            self.index = json.load(f)
        self.labels = pd.read_csv("personality_57.csv")[['hash', 'NEO N', 'NEO E', 'NEO O', 'NEO U', 'NEO S']]
        for column in self.labels.columns:
            if column == 'hash':
                continue
            self.labels[column] = (self.labels[column] - self.labels[column].mean()) / self.labels[column].std()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        info = self.index[idx]
        hash = info["hash"]
        session = info["session"]

        raw = mne.io.read_raw_edf(join(self.root_dir, f"{hash}_rest_standardized_T{session}.edf"))
        segmented = mne.make_fixed_length_epochs(raw=raw, duration=epoch_len, preload=True,
                                                 overlap=epoch_len * overlap)
        data = segmented.get_data()

        big_5_params = self.labels.loc[self.labels['hash'] == hash][['NEO N', 'NEO E', 'NEO O', 'NEO U', 'NEO S']]
        big_5_params = big_5_params.to_numpy()

        sample = [torch.FloatTensor(data),
                  torch.FloatTensor(big_5_params)]  # {'data': data, 'big_5_params': big_5_params}

        if self.transform:
            sample = self.transform(sample)

        return sample


def create_windowed_index():
    # index_path = join(path, 'windowed_index.json')
    index_path = path / 'windowed_index.json'
    if not exists(index_path):
        files = [f for f in listdir(path) if isfile(join(path, f)) and f.lower().endswith(('.edf'))]
        index = []
        for file in tqdm(files):
            raw = mne.io.read_raw_edf(path / file)
            segmented = mne.make_fixed_length_epochs(raw=raw, duration=epoch_len, preload=True,
                                                     overlap=epoch_len * overlap)
            for s in segmented.selection:
                index.append(
                    {"hash": filename_to_patient_id(file), "session": filename_to_session(file), "selection": str(s)})
        json_object = json.dumps(index, indent=4)
        with open(index_path, "w") as outfile:
            outfile.write(json_object)


def create_seq_windowed_index():
    index_path = path / 'seq_windowed_index.json'
    if not exists(index_path):
        files = [f for f in listdir(path) if isfile(join(path, f)) and f.lower().endswith(('.edf'))]
        index = []
        for file in tqdm(files):
            raw = mne.io.read_raw_edf(join(path, file))
            segmented = mne.make_fixed_length_epochs(raw=raw, duration=epoch_len, preload=True,
                                                     overlap=epoch_len * overlap)
            index.append({"hash": filename_to_patient_id(file), "session": filename_to_session(file)})
        json_object = json.dumps(index, indent=4)
        with open(index_path, "w") as outfile:
            outfile.write(json_object)


if __name__ == "__main__":
    # todo zmienne do klas i jako parametry!!!!
    if platform.node().startswith('LAPTOP-0TK'):
        path = Path().absolute() / 'data' / 'REST_standarized'
    elif platform.node().startswith('Igors-MacBook-Pro') or platform.node().startswith('igor-podolak-6.laptop.matinf'):
        DATA_ROOT_DIR = Path('/Users/igor/data')
        # info nazwa kartoteki z plikami -- wartosc w parametrze wywolania --datadir
        #     datadir = f"{DATA_ROOT_DIR}/personality_traits/RESTS_gr87"
        # datadir = DATA_ROOT_DIR / 'personality_traits' / 'RESTS_gr87'
        standardized_dir = DATA_ROOT_DIR / 'personality_traits' / 'RESTS_gr87_standardized'
        path = standardized_dir

    create_windowed_index()
    wed = WindowedEEGDataset(path / 'windowed_index.json', path)
    print(wed.__getitem__(5))
    create_seq_windowed_index()
    seq_wed = WindowedSequenceEEGDataset(path / 'seq_windowed_index.json', path)
    print(seq_wed.__getitem__(5))
