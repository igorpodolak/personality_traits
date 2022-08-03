import argparse
import platform
from pathlib import Path
import mne
import mne.channels
import copy
import pandas as pd
from collections import Counter
import re
from utils import Preprocess
from tqdm import tqdm


# preprocess calej bazy danych
# nowe pliki sa zapisywane w osobnym folderze
# 
def filename_to_patient_id(filename):
    pattern = r'(\w{8})_rest_'
    match = re.search(pattern, Path(filename).parts[-1])
    return match.group(1)


def filename_to_session(filename):
    pattern = r'_rest_T(\d\d?)'
    match = re.search(pattern, filename)
    return match.group(1)


def main(opts):
    pp = Preprocess.Preprocess(person=opts.person, session=opts.session, logdir=Path(opts.logdir),
                               datadir=Path(opts.datadir), filetype=opts.filetype, ica_removal_method="manual",
                               segment_length=2048, reference='average')

    # info read and preprocess: band-pass and notch filter, drop selected channels, interpolate bad channels,
    # info reference to average, segment to overlapping fixed length segments, build morlet epochs
    pp.read_file(drop_channels=True, preload=True)
    before = copy.deepcopy(pp)
    pp.preprocess()

    # path = Path().ab + '\\data\\REST_preprocessed'

    mne.export.export_raw(opts.preproceesed_dir / f'{opts.person}_rest_preprocess_T{opts.session}.edf', pp.raw)
    # pp.raw.save(f'{path}\\{opts.person}_rest_preprocess_T{opts.session}.fif')

    return


if __name__ == '__main__':

    Df = pd.read_csv("./personality_57.csv")

    if platform.node().startswith('LAPTOP-0TK'):
        datadir = Path().absolute() / 'data' / 'REST_baza'
        preprocessed_dir = Path().absolute() / 'REST_preprocessed'
    elif platform.node().startswith('Igors-MacBook-Pro') or platform.node().startswith('igor-podolak-6.laptop.matinf'):
        DATA_ROOT_DIR = Path('/Users/igor/data')
        # info nazwa kartoteki z plikami -- wartosc w parametrze wywolania --datadir
        #     datadir = f"{DATA_ROOT_DIR}/personality_traits/RESTS_gr87"
        datadir = Path(f'{DATA_ROOT_DIR}') / 'personality_traits' / 'RESTS_gr87'
        preprocessed_dir = Path(f'{DATA_ROOT_DIR}') / 'personality_traits' / 'RESTS_gr87_preprocessed'

    channels = []
    failed_patients = []
    raws = []

    for person in Df["hash"]:
        for session in range(1, 21):
            sample_data_raw_file = datadir / person / f"{person}_rest_T{session}.edf"
            try:
                raw = mne.io.read_raw_edf(sample_data_raw_file)
                raws.append(raw)
                channels += raw.ch_names
            except:
                failed_patients.append(person)

    channels_counts = Counter(channels)
    selection = [key for key, value in channels_counts.items() if value == max(channels_counts.values())]

    temp_raws = []
    removed = []
    for raw in raws:
        try:
            r = raw.copy()
            temp_raws.append(raw.pick_channels(selection))
        except:
            removed.append(r)
    raws = temp_raws

    filetype = 'edf'
    logdir = './logdir'
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=datadir, help='Diectory with EDF or SET files')
    parser.add_argument('--preprocessed_dir', type=str, default=preprocessed_dir, help='Diectory with EDF or SET files')
    parser.add_argument('--person', type=str, default='', help='Identifier of file .edf')
    parser.add_argument('--session', type=int, default='', help='number of session')
    parser.add_argument('--filetype', type=str, default=filetype, help='type of file with raw data')
    parser.add_argument('--logdir', type=str, default=logdir, help='Path to directory where save results')

    args = parser.parse_args()

    for raw in tqdm(raws):
        person = filename_to_patient_id(raw.filenames[0])
        session = filename_to_session(raw.filenames[0])
        path_to_file = preprocessed_dir / f'{person}_rest_preprocess_T{session}.edf'
        if path_to_file.is_file():
            pass
        else:
            path_to_file = datadir / person / f'{person}_rest_T{session}.edf'

            args.person = person
            args.session = session
            if path_to_file.is_file():
                main(opts=args)
            else:
                print(f'Nie znaleziono pacjenta {person}')
    print("Finished")
