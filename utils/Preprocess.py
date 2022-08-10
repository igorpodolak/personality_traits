import os
import warnings

import numpy as np
import mne
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from pathlib import Path
import matplotlib.pyplot as plt
from pandas import DataFrame


# def make_mne_info():
#     sfreq = 400
#     ch_types = 19 * ['eeg']
#     ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4',
#                 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
#
#     info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
#     info.set_montage('standard_1005')
#     return info


class Preprocess:

    def __init__(self, person, session, logdir, datadir, filetype, ica_removal_method="manual", sfreq=500,
                 segment_length=2000, t_min=0., t_max=None, reference='average', ica_method="fastica"):
        self.person = person  # name of the patient -- name from file
        self.session = session  # session number if any
        self.logdir = logdir  # logging directory (from call param)
        self.montage_dir = Path(os.path.dirname(mne.__file__)) / 'channels' / 'data' / 'montages'
        self.datadir = datadir  # data dir (from call)
        self.filetype = filetype.lower()
        # self.label_19_order = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4',
        #                        'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        # info channels that are available in EDF recordings
        # self.label_19_order = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
        #                        'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
        # info channels that are available in EEGLAB recordings
        self.label_19_order = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                               'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        # info correct labels
        # self.label_19_order = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
        #                        'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
        self.raw = None  # raw file read
        self.reference = reference
        self.eog_evoked = None  # data evoked (i.e. segmented ???)
        self.ecg_evoked = None
        self.ica = None  # ICA model
        self.ica_method = ica_method  # ICA algorithm to be used., default fastica
        self.ica_removal_method = ica_removal_method  # ICA component removal method
        self.ica_reconstructed = None  # reconstructed from ICA components data
        self.segment_length = segment_length  # length of the segment, def. 2048
        self.epoch_len = 0
        self.segmented = None
        self.freqs = np.arange(1.0, 31.0, 0.5)  # frequencies bins to compute: from 1 to 31 Hz with 0.5 Hz skip
        # self.freqs = np.arange(0.5, 30.5, 0.5)        # frequencies bins to compute: from 1 to 31 Hz with 0.5 Hz skip
        self.morlet = None  # morlet wavelet processed data
        self.baseline_mode = None  # baseline mode for shifting epochs
        self.t_min = t_min
        self.t_max = t_max
        self.sfreq = sfreq

    def read_file(self, drop_channels=True, preload=False):
        if self.filetype == 'edf':
            in_file = Path(self.datadir) / f"{self.person}" / f"{self.person}_rest_T{self.session:02d}.edf"
            self.raw = mne.io.read_raw_edf(input_fname=in_file.absolute().as_posix(), preload=preload)
            # todo resample if sfreq is not 500Hz
            # warning resampling raw file is not optimal -- see MNE resampling
            if self.raw.info['sfreq'] != self.sfreq:
                warnings.warn(f"{in_file} has sfreq == {self.raw.info['sfreq']} != {self.sfreq}; resampling")
                self.raw = self.raw.copy().resample(self.sfreq, npad='auto', n_jobs=3)
            if self.t_max is not None and self.t_min < self.t_max:
                # info crop the raw file to [t_min, t_max) range
                # todo check if the length of file is ok; ?skip file if not???
                # todo not sure whetehr substracting 1 / sfreq was correct...
                # self.raw.crop(tmin=self.t_min, tmax=self.t_max - 1/self.raw.info['sfreq'])
                self.raw.crop(tmin=self.t_min, tmax=self.t_max)
        elif self.filetype == 'fdt':
            in_file = Path(self.datadir) / f"{self.person}" / f"{self.person}_rest_T{self.session}.set"
            self.raw = mne.io.read_raw_eeglab(input_fname=in_file.absolute().as_posix(), eog='auto', preload=preload)
            # self.raw = mne.io.read_epochs_eeglab(input_fname=in_file.absolute().as_posix())
        montage, rename_dict = self.prepare_montage()
        mne.channels.rename_channels(self.raw.info, rename_dict)
        if drop_channels:
            # info drop non-eeg channels
            # todo instead of dropping last three, find some regular expression, e.g. not name.startwith("EEG ")
            to_drop = [x for x in self.raw.ch_names if x not in self.label_19_order]
            # self.drop_channels(channels=self.raw.info['ch_names'][-3:])
            self.drop_channels(channels=to_drop)
        # info save raw file read
        self.raw = self.raw.copy().set_montage(montage=montage, on_missing='ignore')
        # info epoch_len is the segment length divided by sampling frequency
        # info so that segments would not be extended with zeroes
        self.epoch_len = self.segment_length / self.raw.info['sfreq']

    def drop_channels(self, channels=None):
        if channels is None:
            return
        if type(channels) is not tuple and type(channels) is not list:
            channels = [channels]
        to_drop = [ch for ch in channels if ch in self.raw.info.ch_names]
        self.raw = self.raw.copy().drop_channels(to_drop)

    # info prepare a montage for visualization using stadard 1020 setting
    def prepare_montage(self):
        montage = mne.channels.make_standard_montage('standard_1020')
        rename_dict = {x: x.split(' ')[-1] for x in self.raw.ch_names if x.split(' ')[-1] in montage.ch_names}
        sel_dict = mne.channels.make_1020_channel_selections(self.raw.info)
        return montage, rename_dict

    # info do band pass filtering
    def low_high_notch_filter(self, low=0.5, high=30., notch=50., plot=False):
        raw_copy = self.raw.copy()
        if plot:
            self.raw_plot()
        if notch is not None:
            raw_copy = raw_copy.copy().notch_filter((notch, 3 * notch + notch / 5, notch))
            self.raw = raw_copy
            if plot:
                self.raw_plot()
        if low is not None and high is not None:
            raw_copy = raw_copy.copy().filter(low, high, fir_design='firwin')
            self.raw = raw_copy
            if plot:
                self.raw_plot()
        # self.raw = raw_copy

    # info set the reference channels; default is by averaging all
    def set_reference_channels(self, reference='average'):
        if reference == 'average':
            self.raw.set_eeg_reference(ref_channels='average')
            return
        if type(reference) is not tuple and type(reference) is not list:
            reference = [reference]
        for name in reference:
            if name not in self.raw.info["ch_names"]:
                # todo find a nearby channel as reference
                raise ValueError(f"channel {name} is not in raw.info['ch_names]")
        self.raw.set_eeg_reference(ref_channels=reference)

    def raw_plot(self, exclude=None):
        # todo How to set colors?
        if exclude is None:
            exclude = []
        picks = [x for x in self.raw.info["ch_names"] if x.startswith("EEG ")]
        picks = mne.pick_channels(self.raw.info['ch_names'], include=picks, exclude=exclude)
        self.raw.plot_psd(fmin=0., fmax=60., estimate="power", picks=picks)

    def notch_filter(self, freq=50):
        self.raw = self.raw.copy().notch_filter(Fs=self.raw.info['sfreq'], freqs=freq)

    def create_evoked(self):
        self.eog_evoked = create_eog_epochs(self.raw, ch_name="SaO2 SpO2").average()
        self.eog_evoked.apply_baseline(baseline=(None, -0.2))

        self.ecg_evoked = create_ecg_epochs(self.raw, ch_name="HR HR").average()
        self.ecg_evoked.apply_baseline(baseline=(None, -0.2))

    # info do a sort of semi-automatic removal in ica
    def semi_automatic_ica_removal(self):
        self.ica.exclude = []
        # info find which ICs match the EOG pattern
        eog_indices, eog_scores = self.ica.find_bads_eog(self.raw)
        self.ica.exclude = eog_indices

        # info barplot of ICA component "EOG match" scores
        self.ica.plot_scores(eog_scores)

        # info plot diagnostics
        self.ica.plot_properties(self.raw, picks=eog_indices)

        # info plot ICs applied to raw data, with EOG matches highlighted
        self.ica.plot_sources(self.raw, show_scrollbars=False)

        # info plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
        self.ica.plot_sources(self.eog_evoked)

    # info compute raw file ica
    def raw_ica(self, plot=False):
        picks = [x for x in self.raw.info["ch_names"] if x not in self.reference]
        # info visualize the artifacts
        if plot:
            self.raw.plot(n_channels=5, show_scrollbars=True, title="Prior to ICA")
        # if ica_remove_method != 'manual':
        #     self.eog_evoked, self.ecg_evoked = self.create_evoked()

        # info remove slow drifts below 1Hz as may affect ---> ICA solution from filtered may be applied to unfiltered
        filt_raw = self.raw.copy()
        filt_raw.load_data().filter(l_freq=1., h_freq=None)
        # filt_raw.plot(title="raw file highpassed above 1Hz (raw_ica)", show_scrollbars=False)

        # todo ica should probably be done BEFORE removing heartbeat, etc, to remove its impact...
        self.ica = ICA(n_components=len(picks), method=self.ica_method, random_state=97, max_iter=800)
        self.ica.fit(filt_raw, picks=picks)
        if plot:
            f = self.ica.plot_components(
                title=f"ICA components for {self.person} found ({self.ica_method})")  # , show=False)
            # plt.savefig(Path(self.logdir) / "pca_components.png")

        reconst_raw = None
        # info remove some components
        if self.ica_removal_method == 'manual':
            # info we can safely: just remove ica 000 - eye blinks, and ica 001 - heart beat
            self.ica.exclude = [0, 1]
            # ica.plot_properties(raw, picks=ica.exclude)
            # info apply ica removing set components
            reconst_raw = self.raw.copy()
            self.ica.apply(reconst_raw)
            # raw.plot(show_scrollbars=False)

        elif self.ica_removal_method == 'eog':
            self.ica.exclude = []
            # info find which ICs match the EOG pattern
            eog_indices, eog_scores = self.ica.find_bads_eog(self.raw)
            self.ica.exclude = eog_indices
            # info barplot of ICA component "EOG match" scores
            self.ica.plot_scores(eog_scores)
            # info plot diagnostics
            self.ica.plot_properties(self.raw, picks=eog_indices)
            # info plot ICs applied to raw data, with EOG matches highlighted
            self.ica.plot_sources(self.raw, show_scrollbars=False)
            # info plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
            self.ica.plot_sources(self.eog_evoked)
        elif self.ica_removal_method == 'simulated_channel':
            self.ica.exclude = []
            # find which ICs match the ECG pattern
            self.ecg_indices, self.ecg_scores = self.ica.find_bads_ecg(self.raw, method='correlation', threshold='auto')
            self.ica.exclude = self.ecg_indices
            # barplot of ICA component "ECG match" scores
            self.ica.plot_scores(self.ecg_scores)
            # plot diagnostics
            self.ica.plot_properties(self.raw, picks=self.ecg_indices)
            # plot ICs applied to raw data, with ECG matches highlighted
            self.ica.plot_sources(self.raw, show_scrollbars=False)
            # plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
            self.ica.plot_sources(self.ecg_evoked)
        else:
            raise ValueError(f"ica_remove_method {self.ica_removal_method} is not specified")

        self.ica_reconstructed = reconst_raw
        self.raw = reconst_raw.copy()
        if plot:
            self.raw.plot(n_channels=5, show_scrollbars=True, title="After ICA")

    def interpolate_bad_channels(self):
        # todo implement!
        if len(self.raw.info["bads"]) > 0:
            eeg_data = self.raw.copy().pick_types(meg=False, eeg=True, exclude=[])
            eeg_data_interp = eeg_data.copy().interpolate_bads(reset_bads=False)

            for title, data in zip(['orig.', 'interp.'], [eeg_data, eeg_data_interp]):
                fig = data.plot(butterfly=True, color='#00000022', bad_color='r')
                fig.subplots_adjust(top=0.9)
                fig.suptitle(title, size='xx-large', weight='bold')

    def reference_to_average(self, plot=False):
        self.raw = self.raw.copy().set_eeg_reference(ref_channels='average', projection=True)
        if plot:
            self.raw.plot(n_channels=5, show_scrollbars=True, title="Referenced to average")

    def segment_data(self, overlap=0.5, plot=False):
        self.segmented = mne.make_fixed_length_epochs(raw=self.raw, duration=self.epoch_len, preload=True,
                                                      overlap=self.epoch_len * overlap)
        if plot:
            self.segmented.plot(n_channels=5, n_epochs=10, show_scrollbars=True, title="Segmented")
        # segmented.plot_drop_log()

    def morlet_epochs(self, n_jobs=3):
        epochs_loaded = self.segmented.load_data()

        # info set number of cycles for morlet wavelets; less for low frequencies
        n_cycles = np.ones_like(self.freqs) * 7.
        n_cycles[0:4] = [2, 3, 5, 6]

        # epochs_loaded.plot_psd(fmin=0., fmax=35., average=True)
        # epochs_loaded.plot_psd_topomap(normalize=True)
        average_morlet = True
        # warning TUTAJ jest pewnie problem: tfr_morlet() zwraca wartosci
        #  rzedu 10.e-9, podczas gdy powinny byc rzedu 0.1-1.0, a czesem więcej
        mrl_power_avg = mne.time_frequency.tfr_morlet(inst=epochs_loaded, freqs=self.freqs, use_fft=False,
                                                      n_cycles=n_cycles, average=average_morlet, return_itc=False,
                                                      output='power', n_jobs=n_jobs)
        self.baseline_mode = 'ratio'
        mrl_power_avg.apply_baseline(mode=self.baseline_mode, baseline=(-.100, 0))
        self.morlet = mrl_power_avg
        # ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
        #  'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
        if False:
            # to_plot = np.array([['Fp1', 'Fp2'], ['F7', 'F3'], ['Fz', 'F4'], ['F8', 'T7'], ['C3', 'Cz'],
            #                     ['C4', 'T8'], ['P7', 'P3'], ['Pz', 'P4'], ['P8', ''], ['O1', 'O2']])
            # to_plot = np.array([['Fp1', 'F7', 'Fz', 'F8', 'C3', 'C4', 'P7', 'Pz', 'P8', 'O1'],
            #                     ['Fp2', 'F3', 'F4', 'T7', 'Cz', 'T8', 'P3', 'P4', 'O2', 'O2']]).T
            width = 2
            channel_iter = iter(self.raw.ch_names)
            height = len(self.raw.ch_names) // 2 + len(self.raw.ch_names) % 2
            fig, axes = plt.subplots(height, 2, figsize=(8, 6))
            for h in range(height):
                for k in range(width):
                    plot_color_bar = True if k == width - 1 else False
                    try:
                        to_plot = next(channel_iter)
                    except StopIteration:
                        break
                    self.morlet.plot(to_plot, axes=axes[h, k], show=False,
                                     title=f"Morlet spectrum with {self.baseline_mode} baseline", combine=None,
                                     # mode=self.baseline_mode,
                                     vmin=self.morlet.data.min(), vmax=self.morlet.data.max(),
                                     colorbar=False)
            plt.tight_layout()
            plt.show()
        pass

    def morlet_epochs_plot(self):
        # ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
        #  'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
        # to_plot = np.array([['Fp1', 'Fp2'], ['F7', 'F3'], ['Fz', 'F4'], ['F8', 'T7'], ['C3', 'Cz'],
        #                     ['C4', 'T8'], ['P7', 'P3'], ['Pz', 'P4'], ['P8', ''], ['O1', 'O2']])
        # to_plot = np.array([['Fp1', 'F7', 'Fz', 'F8', 'C3', 'C4', 'P7', 'Pz', 'P8', 'O1'],
        #                     ['Fp2', 'F3', 'F4', 'T7', 'Cz', 'T8', 'P3', 'P4', 'O2', 'O2']]).T
        width = 2
        all_to_plot = np.array([['Fp1', 'F7', 'Fz', 'F8', 'Cz', 'P3', 'O1'],
                                ['Fp2', 'F3', 'F4', 'C3', 'C4', 'Pz', 'O2']])
        fig, axes = plt.subplots(all_to_plot.shape[1], width, figsize=(9, 10), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0, wspace=0)
        self.morlet.plot(all_to_plot.reshape(-1), axes=axes.reshape(-1), show=False,
                         title=f"Patient {self.person} Morlet spectrum with {self.baseline_mode} baseline",
                         combine=None,
                         mode=self.baseline_mode, vmin=self.morlet.data.min(), vmax=self.morlet.data.max(),
                         colorbar=False)
        plt.savefig(f"./morlet_epochs_{self.person}.png")
        plt.show()

    def preprocess(self):
        # info read edf file
        self.read_file(drop_channels=True, preload=True)
        # info get 0--30Hz band pass, and remove 50 Hz frequency
        self.low_high_notch_filter(plot=False)
        # info reference electrodes are constant

        # self.raw.plot(show_scrollbars=False)
        # self.set_reference_channels(reference=self.reference)
        # self.raw.plot(show_scrollbars=False)

        # info drop selected channels
        self.drop_channels(channels=self.reference)
        # info do ICA
        # warning ICA procedure should drop eye blinking, muscle movement
        # warning components, but it may be done probably only manually
        # self.raw_ica(plot=True)
        self.interpolate_bad_channels()
        self.reference_to_average(plot=False)
        self.segment_data(plot=False)
        self.morlet_epochs()
        self.morlet_epochs_plot()

    def get_morlet_df(self):
        df = DataFrame(self.morlet.data.mean(axis=-1))
        df.index = self.morlet.ch_names
        col_names = {k: str(x) for k, x in enumerate(self.morlet.freqs)}
        df.rename(columns=col_names, inplace=True)
        return df

    def get_bins_df(self):
        data = self.morlet.data.mean(axis=-1)
        data = (data[:, ::2] + data[:, 1::2]) / 2
        df = DataFrame(data)
        df.index = self.morlet.ch_names
        col_names = {k: str(x) for k, x in enumerate(self.morlet.freqs[::2])}
        df.rename(columns=col_names, inplace=True)

    def morlet_to_csv(self, fname):
        df = self.get_morlet_df()
        df.to_csv(fname)
