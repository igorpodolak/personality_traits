import os
import argparse
import platform
from pathlib import Path
import mne
import mne.channels
import numpy as np

from utils import Preprocess

import matplotlib.pyplot as plt


def main(opts):
    pp = Preprocess.Preprocess(person=opts.person, session=opts.session, logdir=Path(opts.logdir),
                               datadir=Path(opts.datadir), filetype=opts.filetype, ica_removal_method="manual",
                               segment_length=2048, reference='average')

    # info read and preprocess: band-pass and notch filter, drop selected channels, interpolate bad channels,
    # info reference to average, segment to overlapping fixed length segments, build morlet epochs
    pp.preprocess()
    # info get data frame with morlet wavelets
    df = pp.get_morlet_df()
    # info compute frequency bins
    df = pp.get_bins_df()
    return


if __name__ == '__main__':
    if platform.node().startswith('Igors-MacBook-Pro') or platform.node().startswith('igor-podolak-6.laptop.matinf'):
        DATA_ROOT_DIR = '/Users/igor/data/'
    # info nazwa kartoteki z plikami -- wartosc w parametrze wywolania --datadir
    datadir = f"{DATA_ROOT_DIR}/personality_traits/RESTS_gr87"
    # info identyfikator pliku w datadir, e.g. 1CA24B1A/1CA24B1A_rest_T1.edf
    person = "4274BB8B"
    person = "4C369AD1"
    person = "1CA24B1A"
    person = "1DC47E36"
    person = "1B09F58B"
    session = 0
    filetype = 'fdt'
    # filetype = 'edf'
    # person = "20E0F77D"
    logdir = './logdir'

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=datadir, help='Diectory with EDF or SET files')
    parser.add_argument('--person', type=str, default=person, help='Identifier of file .edf')
    parser.add_argument('--session', type=int, default=session, help='number of session')
    parser.add_argument('--filetype', type=str, default=filetype, help='type of file with raw data')
    parser.add_argument('--logdir', type=str, default=logdir, help='Path to directory where save results')

    args = parser.parse_args()

    main(opts=args)
