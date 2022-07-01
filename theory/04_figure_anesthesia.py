# %%
import cmocean as co
import pywt
import matplotlib.gridspec as gridspec
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from methods import WaveletQuantileNormalization, WaveletThresholding
from tools import calculate_metrics, mask_to_intervals

import tools
from tools import filter_bandpass
import scipy.ndimage as ndi
import scipy.signal as ss

import matplotlib.pyplot as plt

plt.style.use("hpub")
plt.rcParams["figure.figsize"] = (20, 6)

# %%


def read_recording(path):
    with h5py.File(path) as f:

        # Meta
        meta = dict(f.attrs)

        # EEG
        freq = f["eeg"].attrs["freq"]
        eeg = pd.DataFrame(
            {
                "time": np.arange(len(f["eeg/eeg1"])) / freq,
                "eeg1": f["eeg/eeg1"][:],
                "eeg2": f["eeg/eeg2"][:],
            }
        )
    return meta, eeg


meta, eeg = read_recording("data/vitaldb/78.h5")


# %% Detect artifacts

signal = np.nan_to_num(eeg.eeg1.values, 0)

hf = filter_bandpass(signal, 32, 63, 128, 4)
power = ndi.gaussian_filter1d(signal**2, 128 // 4)
hf_power = ndi.gaussian_filter1d(hf**2, 128 // 4)

threshold = 6 * np.median(np.sqrt(power)) ** 2

# %%

freq = 128
artifacts = power >= threshold
artifacts = ndi.binary_closing(artifacts, np.ones(128))
artifacts = ndi.binary_opening(artifacts, np.ones(128 // 4))
artifacts = ndi.binary_dilation(artifacts, np.ones(128))

# %%


intervals = tools.mask_to_intervals(artifacts)

times = eeg.loc[:, "time"].values
signal = np.nan_to_num(eeg.loc[:, "eeg1"].values, 0)

wr = WaveletQuantileNormalization("sym5", n=20)
restored = wr.run_single_channel(signal, intervals)


# %% Figure 3

from matplotlib.ticker import EngFormatter

FIG_WIDTH = 2 * 7.16

fs = 1 / eeg.time.diff()[1]

COL_1 = "indianred"
COL_2 = "royalblue"

cmap = "turbo"
nperseg = int(fs * 5)
noverlap = int(fs * 2.5)

freqformatter = EngFormatter(unit="Hz")
timeformatter = EngFormatter(unit="s")
ampformatter = EngFormatter(unit="µV")

plt.close("all")
gs_opts = {
    "height_ratios": [2, 1],
    "hspace": 0.04,
}
fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH / 2))

interval = slice(int(1845 * fs), int((2400) * fs))
f, t, S = ss.spectrogram(signal[interval], nperseg=nperseg, noverlap=noverlap, fs=fs)
t -= t[0]
S = S[(f >= 0) & (f <= 30)]
freqs = f[(f >= 0) & (f <= 30)]
S = S.clip(*np.nanquantile(S, (0.01, 0.99)))

figA, figB = fig.subfigures(ncols=2, wspace=0)
axA1, axA2 = figA.subplots(nrows=2, gridspec_kw=gs_opts, sharex=True)
axA1.pcolormesh(t, freqs, np.log10(np.abs(S) ** 2), cmap=cmap, vmin=-3, shading="auto")
axA1.set_title("A. EEG during general anesthesia", loc="left")
axA1.set_ylabel("Frequency")
axA1.get_xaxis().set_visible(False)
axA1.set_xlim(0, 400)
axA1.yaxis.set_major_formatter(freqformatter)
axA1.yaxis.set_tick_params(rotation=30, direction="out", length=5, labelsize=12)

Δeeg = 150

axA2.plot(times[interval] - times[interval][0], signal[interval], lw=2.5, c="white")
axA2.plot(times[interval] - times[interval][0], signal[interval], lw=0.25, c="#333")
axA2.set_ylabel("EEG", labelpad=-8)
axA2.yaxis.set_major_formatter(ampformatter)
axA2.set_xlabel("Time")
axA2.set_xticks([0, 60, 120, 180, 240, 300, 360])
axA2.set_ylim(-Δeeg, Δeeg)
axA2.xaxis.set_major_formatter(timeformatter)

f, t, S = ss.spectrogram(restored[interval], nperseg=nperseg, noverlap=noverlap, fs=fs)
t -= t[0]

S = S[(f >= 0) & (f <= 30)]
freqs = f[(f >= 0) & (f <= 30)]
S = S.clip(*np.nanquantile(S, (0.005, 0.995)))

axB1, axB2 = figB.subplots(nrows=2, gridspec_kw=gs_opts)
axB1.sharex(axA1)
axB2.sharex(axA1)
axB1.pcolormesh(t, freqs, np.log10(np.abs(S) ** 2), cmap=cmap, vmin=-3, shading="auto")
axB1.set_title("B. Restoration with WQN-transport", loc="left")
axB1.set_ylabel("Frequency")
axB1.get_xaxis().set_visible(False)
axB1.yaxis.set_major_formatter(freqformatter)

axB2.plot(times[interval] - times[interval][0], restored[interval], lw=2.5, c="white")
axB2.plot(times[interval] - times[interval][0], restored[interval], lw=0.25, c="#333")
axB2.set_ylabel("EEG", labelpad=-8)
axB2.set_xlabel("Time")
axB2.set_ylim(-Δeeg, Δeeg)
axB2.yaxis.set_major_formatter(ampformatter)
axB2.xaxis.set_major_formatter(timeformatter)

_artifacts = ndi.binary_dilation(artifacts, np.ones(256))
_artifacts = ndi.binary_closing(_artifacts, np.ones(512))
_artifacts = ndi.binary_opening(_artifacts, np.ones(512))
_art_intervals = tools.mask_to_intervals(
    _artifacts[interval], times[interval] - times[interval][0]
)

for i, j in _art_intervals:
    axA2.axvspan(i, j, color=COL_1, alpha=0.25)
    axB2.axvspan(i, j, color=COL_2, alpha=0.25)

axA1.yaxis.set_tick_params(rotation=30, direction="out", length=5, labelsize=12, pad=0)
axA2.yaxis.set_tick_params(rotation=30, direction="out", length=5, labelsize=12, pad=0)
axB1.yaxis.set_tick_params(rotation=30, direction="out", length=5, labelsize=12, pad=0)
axB2.yaxis.set_tick_params(rotation=30, direction="out", length=5, labelsize=12, pad=0)

fig.subplots_adjust(hspace=0, wspace=0)

fig.savefig("acha_fig4_anesthesia.pdf")


# %%
