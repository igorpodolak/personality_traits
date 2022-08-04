import os
import argparse
import platform
from pathlib import Path
import mne
import mne.channels
import numpy as np
import copy
import pandas as pd
from collections import Counter
import re
from utils import Preprocess
from tqdm import tqdm
from os.path import isfile, join
import json
from utils.file_patient_session import filename_to_patient_id, filename_to_session

mne.set_log_level("WARNING")

Df = pd.read_csv("personality_57.csv")
if platform.node().startswith('LAPTOP-0TK'):
    datadir = Path().absolute() / 'data' / 'REST_preprocessed'
    # preprocessed_dir = Path().absolute() / 'REST_preprocessed'
    path_standarized = Path().absolute() / 'data' / 'REST_standarized'
    savedatadir = Path().absolute() / "data"
    mean_file_path = Path(savedatadir) / 'means_of_channels.txt'
    std_file_path = Path(savedatadir) / 'std_of_channels.txt'
elif platform.node().startswith('Igors-MacBook-Pro') or platform.node().startswith('igor-podolak-6.laptop.matinf'):
    DATA_ROOT_DIR = Path('/Users/igor/data')
    datadir = DATA_ROOT_DIR / 'personality_traits' / 'RESTS_gr87_preprocessed'
    path_standardized = DATA_ROOT_DIR / 'personality_traits' / 'RESTS_gr87_standardized'
    savedatadir = Path().absolute() / "data"
    mean_file_path = Path(savedatadir) / 'means_of_channels.txt'
    std_file_path = Path(savedatadir) / 'std_of_channels.txt'
else:
    print(f"Unknown computer {platform.node()}, exiting")
    exit()

files = [f for f in os.listdir(datadir) if isfile(join(datadir, f)) if not f.startswith('.DS_Store')]

os.chdir(datadir)

channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4',
                 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

list_of_summed_channels = dict.fromkeys(channel_names, 0)
list_of_summed_channels_len = dict.fromkeys(channel_names, 0)

list_of_means = dict.fromkeys(channel_names, 0)

if mean_file_path.is_file():
    pass
else:
    for file_str in tqdm(files):
        file = mne.io.read_raw_edf(file_str)
        for channel in channel_names:
            if channel not in file.ch_names:
                print(f"{channel} is not present in {file_str}")
                exit(1)
            list_of_summed_channels[channel] += np.sum(file.get_data(picks=[channel]))  # append sum of channel in file
            list_of_summed_channels_len[channel] += len(file.get_data(picks=[channel])[0])  # append len of channel data
            pass

    for channel in channel_names:
        list_of_means[channel] = list_of_summed_channels[channel] / list_of_summed_channels_len[channel]

    os.chdir(savedatadir)
    with open("means_of_channels.txt", "w+") as f:
        dict_of_means = json.dump(list_of_means, f)

list_of_subtractions_squared = dict.fromkeys(channel_names, 0)
channels_full_len = dict.fromkeys(channel_names, 0)
list_of_std = dict.fromkeys(channel_names, 0)

os.chdir(savedatadir)
with open("means_of_channels.txt", "r") as f:
    dict_of_means = json.load(f)

os.chdir(datadir)
if std_file_path.is_file():
    pass
else:
    for file_str in tqdm(files):
        file = mne.io.read_raw_edf(file_str)
        for channel in channel_names:
            subtraction_squared = np.sum(np.square(
                file.get_data(picks=[channel]) - np.ones(len(file.get_data(picks=[channel]))) * dict_of_means[channel]))
            list_of_subtractions_squared[channel] += subtraction_squared  # append of sum of subs sqred / n for one file
            channels_full_len[channel] += len(file.get_data(picks=[channel])[0])
    for channel in channel_names:
        list_of_std[channel] = np.sqrt(
            list_of_subtractions_squared[channel] / channels_full_len[channel])  # std for all patiens for channel i

    os.chdir(savedatadir)
    with open('std_of_channels.txt', 'w+') as f:
        json.dump(list_of_std, f)

os.chdir(savedatadir)
with open("std_of_channels.txt", "r") as f:
    dict_of_std = json.load(f)

os.chdir(datadir)
arrays = [0] * 19

for file_str in tqdm(files):
    file_path = Path(
        path_standardized) / f'{filename_to_patient_id(file_str)}_rest_standardized_T{filename_to_session(file_str)}.edf'
    if file_path.is_file():
        pass
    else:
        file = mne.io.read_raw_edf(file_str)
        for i, channel in enumerate(channel_names):
            arrays[i] = (file.get_data(picks=[channel][0]) - np.ones(len(file.get_data(picks=[channel])[0])) *
                         dict_of_means[channel]) / dict_of_std[channel]
        new_array = mne.io.RawArray(np.squeeze(arrays), info=file.info)
        mne.export.export_raw(file_path, new_array)

print('Finished!')
