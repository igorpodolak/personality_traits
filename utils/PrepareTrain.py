from itertools import product
from copy import copy
from pathlib import Path
import numpy as np
import pandas as pd
import re


# import matplotlib.pyplot as plt


class PrepareTrain:

    def __init__(self, datadir, state, eyes, surveyfile, logdir, channels=64):
        self._min_frq: int = 0
        self._max_frq: int = 0
        self.datadir = datadir
        self.state = state if type(state) in [list, tuple] else [state]
        self.eyes = eyes if type(eyes) in [list, tuple] else [eyes]
        self.survey = pd.read_csv(surveyfile)
        self.logdir = logdir
        self.channels = channels

        self.correct_name_re = re.compile(r"^[0-9]*(pre|post)_Average_E[OC].csv$")
        self.subj_re = re.compile(r"^([0-9]+)")

        self.channel_64_order = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
                                 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                                 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2',
                                 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz',
                                 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
                                 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
        # info channel_19_order to etykiety wybrane z Jach by pasowały do naszych
        self.channel_19_order = ['Fp1', 'F3', 'F7', 'C3', 'T7', 'P3', 'P7', 'O1', 'Pz', 'Fp2',
                                 'Fz', 'F4', 'F8', 'Cz', 'C4', 'T8', 'P4', 'P8', 'O2']
        self.rogala_order_orig = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                                  'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        # info ponizej niektore kanaly sa podstawione: T3 --> T7, T4 --> T8, T5 --> P7, T6 --> P8
        self.rogala_order = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
                             'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
        if self.channels == 64:
            self.channel_order = self.channel_64_order
        elif self.channels == 19:
            self.channel_order = self.channel_19_order
        else:
            raise ValueError(f"Number of channels set to {self.channels} is incorrect")

        self.bfas_names = ["Conscientiousness", "Agreeableness", "Neuroticism", "OpenIntell", "Extraversion"]
        self.bfas_short = ["C", "A", "N", "O", "E"]
        self.bfas_rename = {trait: trait_short for trait, trait_short in zip(self.bfas_names, self.bfas_short)}

        self.df = None
        self.df_train = None

    def read_data(self):
        df_all = pd.DataFrame()
        # todo move to select_freq_bin read survey file
        # corr_name_re = re.compile(r"^[0-9]*p[or][se]t?_Average_E[OC].csv$")
        tid = 0
        data = None
        for state, eyes in product(self.state, self.eyes):
            for fn in Path(f"{self.datadir}/{state}/{eyes}").iterdir():
                if self.correct_name_re.match(fn.name) is None:
                    continue
                df = pd.read_csv(fn, header=None, index_col=0)
                index_orig = df.index
                # info get 1Hz bins as means
                data = df.to_numpy()
                data = (data[:, ::2] + data[:, 1::2]) / 2
                df = pd.DataFrame(data)
                df.index = index_orig
                freqs = {k: k + 1 for k in np.arange(0, data.shape[1])}
                df.rename(columns=freqs, inplace=True)
                subject = int(self.subj_re.match(fn.stem)[0])
                # todo move adding bfas values to select
                # bfas = self.survey[self.survey['subject'] == subject][self.bfas_names]
                # for name, short in zip(self.bfas_names, self.bfas_short):
                #     df[short] = bfas[name].to_numpy()[0]
                df.insert(0, column='eyes', value=eyes)
                df.insert(0, column='state', value=state)
                df.insert(0, column='id', value=tid)
                df.insert(0, column='subject', value=subject)
                tid += 1
                if df.shape[0] != 64:
                    print(f"Wrong number of channels: {df.shape[1]}!")
                df_all = pd.concat([df_all, df])
        self.df = df_all
        self.min_frq = 1
        self.max_frq = data.shape[1]

    # info read a single rogala file -- a very primitive version
    def read_single_rogala_power_data(self, fn, tid=0):
        # df_all = pd.DataFrame()
        # todo move to select_freq_bin read survey file
        # corr_name_re = re.compile(r"^[0-9]*p[or][se]t?_Average_E[OC].csv$")
        # id = 0
        # df = pd.read_csv(fn, header=None, index_col=0)
        df = pd.read_csv(fn, header=None, index_col=0)  # info index in column == 0
        # index_orig = df.index  # info unknown order, setting the one from class
        # info get 1Hz bins as means
        data = df.to_numpy()
        data = (data[:, ::2] + data[:, 1::2]) / 2
        df = pd.DataFrame(data)
        df.index = self.rogala_order
        # info change order to that inherited from Jach data
        df = df.reindex(self.channel_19_order)
        freqs = {k: k + 1 for k in np.arange(0, data.shape[1])}
        df.rename(columns=freqs, inplace=True)
        # todo ????
        # subject = int(self.subj_re.match(fn.stem)[0])
        # todo move adding bfas values to select
        # bfas = self.survey[self.survey['subject'] == subject][self.bfas_names]
        # for name, short in zip(self.bfas_names, self.bfas_short):
        #     df[short] = bfas[name].to_numpy()[0]
        df.insert(0, column='eyes', value="unknown")  # todo unknown value yet
        df.insert(0, column='state', value="unknown")  # todo unknown value yet
        df.insert(0, column='id', value=tid)  # todo  unknown yet
        df.insert(0, column='subject', value=0)  # todo unknown yet, if any
        # id += 1
        # if df.shape[0] != 64:
        #     print(f"Wrong number of channels: {df.shape[1]}!")
        # df_all = pd.concat([df_all, df])
        # self.df = df_all
        # self.min_frq = 1
        # self.max_frq = data.shape[1]
        # todo return dataframe or numpy array?
        return df

    # info read a single rogala file -- a very primitive version
    def read_several_rogala_power_data(self, dirname):
        df_all = pd.DataFrame()
        for tid, file in enumerate(Path(dirname).glob('*.csv')):
        # for tid, file in sorted(Path(dirname).glob('*.csv'),
        #                         key=lambda path: int(path.stem.rsplit("_", 1)[1][1:])):
        #     print(file)
            exmpl = self.read_single_rogala_power_data(file, tid=tid)
            df_all = pd.concat([df_all, exmpl])
        return df_all

    def select_freq_bin(self, frq):
        cols = ['subject', 'id', frq]
        # cols.extend(bfas_short)
        df_short = self.df[cols]
        df_short.insert(0, column="channel", value=df_short.index)
        df_all = pd.DataFrame()

        for subject in df_short["subject"].unique():
            df_subj = df_short[df_short['subject'] == subject]
            dft = df_subj[df_subj['subject'] == subject].drop(columns=['subject']).pivot('id', 'channel')
            dft.columns = dft.columns.levels[1]
            dft = dft[self.channel_order]
            if dft.shape[1] != self.channels:
                print(f"Incorrect number of channels: {dft.shape[1]}, should be {self.channels}!")
            col_ren = {col: f"{col}_{frq}" for col in dft.columns}
            dft.rename(columns=col_ren, inplace=True)
            bfas = self.survey[self.survey['subject'] == subject][self.bfas_names]
            bfas.rename(columns=self.bfas_rename, inplace=True)
            # dft = pd.concat([dft, bfas], axis=1)
            dft[bfas.columns] = bfas.iloc[0]
            dft.insert(0, column='subject', value=subject)
            df_all = pd.concat([df_all, dft])
            pass

        self.df_train = df_all

    def get_X_y(self, predict_label):
        y_predict = self.df_train[predict_label].to_numpy()
        df = self.df_train.copy()
        to_drop = copy(self.bfas_short)
        to_drop.append('subject')
        df.drop(columns=to_drop, inplace=True)
        x_var = df.to_numpy()

        return x_var, y_predict

    @property
    def min_frq(self):
        return self._min_frq

    @min_frq.setter
    def min_frq(self, value):
        self._min_frq = value

    @property
    def max_frq(self):
        return self._max_frq

    @max_frq.setter
    def max_frq(self, value):
        self._max_frq = value
