import pywt
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

plt.style.use("resources/figstyle.mplstyle")

FIG_WIDTH = 2.3 * 7.16


# Load data and normalize
eeg = np.load("resources/data/eeg-denoise-net/EEG_all_epochs.npy")
eog = np.load("resources/data/eeg-denoise-net/EOG_all_epochs.npy")
emg = np.load("resources/data/eeg-denoise-net/EMG_all_epochs.npy")

n_eeg = (eeg - eeg.mean(axis=-1).reshape(-1, 1)) / eeg.std(axis=-1).reshape(-1, 1)
n_eog = (eog - eog.mean(axis=-1).reshape(-1, 1)) / eog.std(axis=-1).reshape(-1, 1)
n_emg = (emg - emg.mean(axis=-1).reshape(-1, 1)) / emg.std(axis=-1).reshape(-1, 1)


# Wavelet transform

max_levels = 5
eeg_coeffs = pywt.wavedec(
    n_eeg, "sym5", level=max_levels, axis=-1, mode="periodization"
)
eog_coeffs = pywt.wavedec(
    n_eog, "sym5", level=max_levels, axis=-1, mode="periodization"
)
emg_coeffs = pywt.wavedec(
    n_emg, "sym5", level=max_levels, axis=-1, mode="periodization"
)


# %%

fig, ax = plt.subplots()


ax.plot([cs.var() for cs in eeg_coeffs[1:]], marker="o", label="EEG")

ax.plot([cs.var() for cs in eog_coeffs[1:]], marker="o", label="EOG")

ax.plot([cs.var() for cs in emg_coeffs[1:]], marker="o", label="EMG")

ax.set_ylabel("Variance of wavelet coefficients")
ax.set_xticks([0, 1, 2, 3, 4])

ax.set_xticklabels(["cD5", "cD4", "cD3", "cD2", "cD1"])

ax.set_yscale("log")

ax.legend()

fig.savefig("../../2022_WQN/fig_variance_vs_scales.svg")

# %%
