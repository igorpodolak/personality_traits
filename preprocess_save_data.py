import argparse
import platform
import warnings
from pathlib import Path
import mne
import mne.channels
import copy
import pandas as pd
from collections import Counter
import re
from utils import Preprocess
from tqdm import tqdm
from utils.file_patient_session import filename_to_patient_id, filename_to_session


# preprocess calej bazy danych
# nowe pliki sa zapisywane w osobnym folderze

def main(opts):
    pp = Preprocess.Preprocess(person=opts.person, session=opts.session, logdir=Path(opts.logdir),
                               datadir=Path(opts.datadir), filetype=opts.filetype, ica_removal_method="manual",
                               segment_length=2000, reference='average', t_min=30, t_max=230.)

    # info read and preprocess: band-pass and notch filter, drop selected channels, interpolate bad channels,
    # info reference to average, segment to overlapping fixed length segments, build morlet epochs
    # pp.read_file(drop_channels=True, preload=True)
    # before = copy.deepcopy(pp)
    pp.preprocess(plot=False)

    # path = Path().ab + '\\data\\REST_preprocessed'

    mne.export.export_raw(opts.preprocessed_dir / f'{opts.person}_rest_preprocess_T{opts.session:02d}.edf', pp.raw)
    # pp.raw.save(f'{path}\\{opts.person}_rest_preprocess_T{opts.session}.fif')

    return


if __name__ == '__main__':

    Df = pd.read_csv("./personality_57_rec.csv")

    if platform.node().startswith('LAPTOP-0TK'):
        datadir = Path().absolute() / 'data' / 'REST_baza'
        preprocessed_dir = Path().absolute() / 'REST_preprocessed'
    elif platform.node().startswith('Igors-MacBook-Pro') or platform.node().startswith('igor-podolak-6.laptop.matinf'):
        DATA_ROOT_DIR = Path('/Users/igor/data')
        # info nazwa kartoteki z plikami -- wartosc w parametrze wywolania --datadir
        #     datadir = f"{DATA_ROOT_DIR}/personality_traits/RESTS_gr87"
        datadir = DATA_ROOT_DIR / 'personality_traits' / 'RESTS_gr87'
        preprocessed_dir = DATA_ROOT_DIR / 'personality_traits' / 'RESTS_gr87_preprocessed'

    channels = []
    failed_patients = []
    raws = []

    for person in Df["hash"]:
        for session in range(1, 21):
            sample_data_raw_file = datadir / person / f"{person}_rest_T{session:02d}.edf"
            try:
                raw = mne.io.read_raw_edf(sample_data_raw_file)
                raws.append(raw)
                # info sampling has to be 500 Hz everywhere, if not a resampling is done here
                # warning resampling of a raw file carries with it some artefacts (read MNE docs)
                if raw.info['sfreq'] != 500:
                    warnings.warn(f"{sample_data_raw_file} has sfreq == {raw.info['sfreq']} != 500; resampling")
                    raw_500 = raw.copy().resample(500, npad='auto', n_jobs=3)
                    raw = raw_500
                    mne.export.export_raw(fname=sample_data_raw_file, raw=raw, overwrite=True)
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
    parser.add_argument('--person', type=str, default=None, help='Identifier of file .edf')
    parser.add_argument('--session', type=int, default=None, help='number of session')
    parser.add_argument('--filetype', type=str, default=filetype, help='type of file with raw data')
    parser.add_argument('--logdir', type=str, default=logdir, help='Path to directory where save results')

    args = parser.parse_args()

    # todo we shall skip session 1 from all recordings; do it here or in later preprocessing steps?
    for raw in tqdm(raws):
        person = filename_to_patient_id(raw.filenames[0])
        session = filename_to_session(raw.filenames[0])
        # info skip session 01 recordings
        if session == 1:
            continue
        path_to_file = preprocessed_dir / f'{person}_rest_preprocess_T{session:02d}.edf'
        if path_to_file.is_file():
            pass
        else:
            path_to_file = datadir / person / f'{person}_rest_T{session:02d}.edf'
            args.person = person
            args.session = session
            if path_to_file.is_file():
                main(opts=args)
            else:
                print(f'Nie znaleziono pacjenta {person}')

    print("Finished")
