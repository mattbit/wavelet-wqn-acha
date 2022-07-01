from tools import calculate_snr
from tools import mask_to_intervals, intervals_to_mask
import wfdb
import h5py
import scipy.io
import numpy as np
from pathlib import Path
import scipy.signal as ss
import scipy.ndimage as ndi
import scipy.io

from tools import filter_bandpass

import matplotlib.pyplot as plt

plt.style.use("hpub")
plt.rcParams["figure.figsize"] = (15, 6)


# %%

output_path = Path("data/mfa")
dataset_path = Path("data/physiobank-motion-artifacts")

record_name = "eeg_2"
record_path = str(dataset_path.joinpath(record_name))

record = wfdb.rdrecord(record_path)

fs = record.fs
eeg = record.p_signal[:, :2]

ts = np.arange(eeg.shape[0]) / fs



eeg_ref = eeg[:, 0]
eeg_art = eeg[:, 1]
slice_ = slice(400 * fs, 460 * fs)



# scipy.io.savemat(
#     output_path.joinpath(f"motion_{record_name}.mat"),
#     {
#         "eeg_reference": eeg[:, 0],
#         "eeg_artifact": eeg[:, 1],
#     },
# )

# %%


fig3, ax3 = plt.subplots()
ax3.plot(ts[slice_], eeg_ref[slice_] - eeg_ref[slice_].mean() + 0.5)
ax3.plot(ts[slice_], eeg_art[slice_] - eeg_art[slice_].mean())
fig3.savefig("figures/mfa_fig1/signal.eps")
fig3.savefig("figures/mfa_fig1/signal.pdf")
fig3.savefig("figures/mfa_fig1/signal.svg")

fig3z, ax3z = plt.subplots()
ax3z.plot(ts[slice_][:5 * fs], eeg_ref[slice_][:5 * fs])
fig3z.savefig("figures/mfa_fig1/signal_zoom_5seconds.eps")
fig3z.savefig("figures/mfa_fig1/signal_zoom_5seconds.pdf")
fig3z.savefig("figures/mfa_fig1/signal_zoom_5seconds.svg")


# %%

f, S_ref = ss.welch(eeg_ref, fs=fs, nperseg=fs * 5)
f, S_art = ss.welch(eeg_art, fs=fs, nperseg=fs * 5)

fig4, ax4 = plt.subplots()
ax4.plot(f, S_ref, label="Normal EEG")
ax4.plot(f, S_art, label="Artifacted EEG", c='r')
ax4.set_xlim(1e-1, 500)
ax4.set_yscale("log")
ax4.set_xscale("log")

fig4.savefig("figures/mfa_fig1/signal_spectrum.eps")
fig4.savefig("figures/mfa_fig1/signal_spectrum.pdf")
fig4.savefig("figures/mfa_fig1/signal_spectrum.svg")


# %%

from tools import filter_bandpass, filter_highpass

signal = eeg[:, 0]

# data = filter_bandpass(signal, 0, 5, fs, 4)
data = signal - filter_highpass(signal, 2, fs, 4)
# data = signal

scipy.io.savemat(output_path.joinpath(f"{record_name}_test.mat"), {"data": data})

f, S = ss.welch(signal, fs=fs, nperseg=fs * 5)
ff, Sf = ss.welch(data, fs=fs, nperseg=fs * 5)

plt.plot(f, S)
# plt.plot(ff, Sf)
plt.yscale("log")
plt.xscale("log")
# plt.xlim(0, 100)

# %%

# for q, structure_fun, slope, intercept in zip(qs, struct_funs, slopes, intercepts):
#     plt.plot(np.log2(scale), structure_fun)
#     plt.plot(np.arange(j1, j2 + 1), intercept + np.arange(j1, j2 + 1) * slope)
#     plt.plot(np.log2(scale), intercept + np.log2(scale) * slope, ls="--")
#     plt.title(f"q = {q}")
#     plt.show()

# %%

mfa = scipy.io.loadmat("results/mfa/mfa_motion_eeg_2.mat")

struct_funs = mfa["logstat"]["DWT"][0, 0]["est"][0, 0]
qs = mfa["logstat"]["DWT"][0, 0]["param_est"][0, 0]["q"][0, 0][0]
j1 = mfa["logstat"]["DWT"][0, 0]["param_est"][0, 0]["j1"][0, 0][0, 0]
j2 = mfa["logstat"]["DWT"][0, 0]["param_est"][0, 0]["j2"][0, 0][0, 0]
scale = mfa["logstat"]["DWT"][0, 0]["scale"][0, 0][0]

slopes = mfa["est"]["DWT"][0, 0]["t"][0, 0][0]
intercepts = mfa["est"]["DWT"][0, 0]["aest"][0, 0][0]

h_min = mfa["logstat"]["DWT"][0, 0]["supcoef"][0, 0][0]
h_min_slope = mfa["est"]["DWT"][0, 0]["h_min"][0, 0][0, 0]
h_min_intercept = mfa["est"]["DWT"][0, 0]["h_min_aest"][0, 0][0, 0]


iq = np.argmax(qs >= 1)

fig1, ax1 = plt.subplots()

ax1.axvspan(j1 - 0.5, j2 + 0.5, fc="lightcyan")

ax1.plot(np.log2(scale), struct_funs[iq], marker="o", label="Clean EEG")
ax1.plot(np.arange(j1, j2 + 1), intercepts[iq] + np.arange(j1, j2 + 1) * slopes[iq])
ax1.plot(np.log2(scale), intercepts[iq] + np.log2(scale) * slopes[iq], ls="--")
ax1.set_title(f"q = {round(qs[iq], 2)}")


fig2, ax2 = plt.subplots()
ax2.axvspan(j1 - 0.5, j2 + 0.5, fc="lightcyan")

ax2.plot(np.log2(scale), np.log2(h_min), marker="o")
ax2.plot(np.arange(j1, j2 + 1), h_min_intercept + np.arange(j1, j2 + 1) * h_min_slope)
ax2.plot(np.log2(scale), h_min_intercept + np.log2(scale) * h_min_slope, ls="--")
ax2.set_title("h_min")

mfa = scipy.io.loadmat("results/mfa/mfa_motion_eeg_2_artifact.mat")

struct_funs = mfa["logstat"]["DWT"][0, 0]["est"][0, 0]
qs = mfa["logstat"]["DWT"][0, 0]["param_est"][0, 0]["q"][0, 0][0]
j1 = mfa["logstat"]["DWT"][0, 0]["param_est"][0, 0]["j1"][0, 0][0, 0]
j2 = mfa["logstat"]["DWT"][0, 0]["param_est"][0, 0]["j2"][0, 0][0, 0]
scale = mfa["logstat"]["DWT"][0, 0]["scale"][0, 0][0]

slopes = mfa["est"]["DWT"][0, 0]["t"][0, 0][0]
intercepts = mfa["est"]["DWT"][0, 0]["aest"][0, 0][0]

h_min = mfa["logstat"]["DWT"][0, 0]["supcoef"][0, 0][0]
h_min_slope = mfa["est"]["DWT"][0, 0]["h_min"][0, 0][0, 0]
h_min_intercept = mfa["est"]["DWT"][0, 0]["h_min_aest"][0, 0][0, 0]


ax1.plot(np.log2(scale), struct_funs[iq], marker="o", label="Artifacted EEG", c='r')
# ax.plot(np.arange(j1, j2 + 1), intercepts[iq] + np.arange(j1, j2 + 1) * slopes[iq])
# ax.plot(np.log2(scale), intercepts[iq] + np.log2(scale) * slopes[iq], ls="--")


ax2.plot(np.log2(scale), np.log2(h_min), marker="o", c='r')



fig1.savefig("figures/mfa_fig1/structure_function_q1.eps")
fig1.savefig("figures/mfa_fig1/structure_function_q1.pdf")
fig1.savefig("figures/mfa_fig1/structure_function_q1.svg")


fig2.savefig("figures/mfa_fig1/mfa_h_min.eps")
fig2.savefig("figures/mfa_fig1/mfa_h_min.pdf")
fig2.savefig("figures/mfa_fig1/mfa_h_min.svg")
