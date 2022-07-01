# %%
import pywt
import numpy as np

import matplotlib.pyplot as plt

plt.style.use("hpub")
plt.rcParams["figure.figsize"] = (12, 8)

# %%

eeg = np.load('data/eeg-denoise-net/EEG_all_epochs.npy')
eog = np.load('data/eeg-denoise-net/EOG_all_epochs.npy')
emg = np.load('data/eeg-denoise-net/EMG_all_epochs.npy')

# %%

def remove_artifact(reference, artifact):
    cs_signal = pywt.wavedec(reference, "sym5", mode="periodization")
    cs_artifact = pywt.wavedec(artifact, "sym5", mode="periodization")

    coeffs = [c.copy() for c in cs_artifact]
    for cs_s, cs in zip(cs_signal, coeffs):
        order = np.argsort(np.abs(cs))
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

        vals_ref = np.abs(cs_s)
        ref_order = np.argsort(vals_ref)
        ref_sp = np.linspace(0, len(inv_order) - 1, len(ref_order))
        vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

        r = vals_norm / np.abs(cs)
        cs[:] *= np.minimum(1, r)

    rec = pywt.waverec(coeffs, "sym5", mode="periodization")
    return rec

# %%

artifact = emg[438]
signal = eeg[218]
artifact /= artifact.std()
signal /= signal.std()
artifacted_signal = signal + artifact

corrected = remove_artifact(signal, artifacted_signal)

plt.plot(signal)
plt.plot(artifacted_signal)
plt.plot(corrected)

# %%

signal = eeg[467]
signal_artifacted = signal + eog[260]

cs_signal = pywt.wavedec(signal, "sym5")
cs_artifact = pywt.wavedec(signal_artifacted, "sym5")

coeffs = [c.copy() for c in cs_artifact]
for cs_s, cs in zip(cs_signal, coeffs):
    order = np.argsort(np.abs(cs))
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))

    vals_ref = np.abs(cs_s)
    ref_order = np.argsort(vals_ref)
    ref_sp = np.linspace(0, len(inv_order) - 1, len(ref_order))
    vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

    r = vals_norm / np.abs(cs)
    cs[:] *= np.minimum(1, r)

cs_reconstruction = coeffs

rec = pywt.waverec(coeffs, "sym5")

# %%

plt.plot(signal_artifacted, alpha=0.1, label="Artifacted signal")
plt.plot(signal, label="True signal")
plt.plot(rec, label="Reconstruction")
plt.legend()

# plt.savefig("figures/theory/example_of_signal_eog.svg")
# plt.savefig("figures/theory/example_of_signal_eog.eps")

# %%

num_levels = len(cs_signal) - 1
level_labels = [f"cA{num_levels}"] + [f"cD{n}" for n in range(num_levels, 0, -1)]

fig, axes = plt.subplots(nrows=num_levels + 1, ncols=2, figsize=(10, 12))

ts = np.arange(signal.size) / 256

for n, (cs_ref, cs_art, cs_rec, level_label) in enumerate(zip(cs_signal, cs_artifact, cs_reconstruction, level_labels)):
    cs_ref_abs = np.abs(cs_ref)
    cs_art_abs = np.abs(cs_art)
    cs_rec_abs = np.abs(cs_rec)

    p = np.linspace(0, 1, cs_ref_abs.size)

    axes[n, 0].plot(sorted(cs_ref_abs), p, label="Reference signal")
    axes[n, 0].plot(sorted(cs_art_abs), p, label="Artifacted")
    axes[n, 0].plot(sorted(cs_rec_abs), p, label="Reconstruction")
    axes[n, 0].set_ylabel(level_label)

    cs_proj_rec = [np.zeros_like(cs_) for cs_ in cs_reconstruction]
    cs_proj_art = [np.zeros_like(cs_) for cs_ in cs_reconstruction]
    cs_proj_ref = [np.zeros_like(cs_) for cs_ in cs_reconstruction]

    cs_proj_rec[n][:] = cs_rec
    cs_proj_art[n][:] = cs_art
    cs_proj_ref[n][:] = cs_ref

    proj_rec = pywt.waverec(cs_proj_rec, "sym5")
    proj_ref = pywt.waverec(cs_proj_ref, "sym5")
    proj_art = pywt.waverec(cs_proj_art, "sym5")

    axes[n, 1].plot(ts, proj_ref, label="Reference signal")
    axes[n, 1].plot(ts, proj_art, label="Artifacted")
    axes[n, 1].plot(ts, proj_rec, label="Reconstruction")

    axes[n, 0].sharex(axes[0, 0])
    axes[n, 1].sharex(axes[0, 1])
    axes[n, 0].xaxis.set_visible(False)
    axes[n, 1].xaxis.set_visible(False)


axes[0, 0].legend()
axes[-1, 0].xaxis.set_visible(True)
axes[-1, 1].xaxis.set_visible(True)
axes[-1, 0].set_xlabel("Amplitude")
axes[-1, 1].set_xlabel("Time (s)")

# fig.savefig(f"figures/theory/example_of_transport_eog.svg")
# fig.savefig(f"figures/theory/example_of_transport_eog.eps")



# %%

import scipy.signal as ss

freq, S_eeg = ss.welch(n_eeg, fs=256, nperseg=512, noverlap=128)
freq, S_eog = ss.welch(n_eog, fs=256, nperseg=512, noverlap=128)
freq, S_emg = ss.welch(n_emg, fs=256, nperseg=512, noverlap=128)

fig, ax = plt.subplots()
ax.plot(freq, S_eeg.mean(axis=0), label="EEG")
ax.plot(freq, S_eog.mean(axis=0), label="EOG")
ax.plot(freq, S_emg.mean(axis=0), label="EMG")
ax.set_xlim(0, 40)
ax.set_xlabel("Frequency (Hz)")
ax.legend()

fig.savefig("figures/theory/psd_by_type.svg")
