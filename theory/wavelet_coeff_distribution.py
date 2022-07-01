# %%
import pywt
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

plt.style.use("hpub")
plt.rcParams["figure.figsize"] = (12, 8)

# %%

from scipy import optimize


def fit_generalized_gaussian(x):
    μ0 = x.mean()
    σ0 = x.std()
    β0 = 2

    res = optimize.minimize(
        neg_gg_likelihood,
        (μ0, σ0, β0),
        args=(x,),
        bounds=[(-np.inf, np.inf), (1e-2, np.inf), (1e-2, np.inf)]
    )

    return res.x




def neg_gg_likelihood(θ, x):
    μ, σ, β = θ
    return -stats.gennorm.logpdf(x, loc=μ, scale=σ, beta=β).sum()


# %%

eeg = np.load("data/eeg-denoise-net/EEG_all_epochs.npy")
eog = np.load("data/eeg-denoise-net/EOG_all_epochs.npy")
emg = np.load("data/eeg-denoise-net/EMG_all_epochs.npy")

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

    cs_reconstruction = coeffs

    rec = pywt.waverec(coeffs, "sym5", mode="periodization")
    return rec


# %% Timeseries examples
ts = np.arange(eeg.shape[1]) / 256

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True)

ax0.plot(ts, eeg[9])
ax0.set_yticks([])
ax0.set_ylabel("EEG")

ax1.plot(ts, eog[9])
ax1.set_yticks([])
ax1.set_ylabel("EOG")

ax2.plot(ts, emg[9])
ax2.set_yticks([])
ax2.set_ylabel("EMG")

ax2.set_xticks([0, 0.5, 1, 1.5, 2])
ax2.set_xlabel("Time (s)")
# fig.savefig("figures/theory/timeseries_eeg_eog_emg.svg")
# fig.savefig("figures/theory/timeseries_eeg_eog_emg.eps")

# %%

n_eeg = (eeg - eeg.mean(axis=-1).reshape(-1, 1)) / eeg.std(axis=-1).reshape(-1, 1)
n_eog = (eog - eog.mean(axis=-1).reshape(-1, 1)) / eog.std(axis=-1).reshape(-1, 1)
n_emg = (emg - emg.mean(axis=-1).reshape(-1, 1)) / emg.std(axis=-1).reshape(-1, 1)

mix = n_eeg + n_emg[: len(n_eeg)]
fixed = np.array([remove_artifact(ref, art) for ref, art in zip(n_eeg, mix)])

max_levels = 4
eeg_coeffs = pywt.wavedec(
    n_eeg, "sym5", level=max_levels, axis=-1, mode="periodization"
)
eog_coeffs = pywt.wavedec(
    n_eog, "sym5", level=max_levels, axis=-1, mode="periodization"
)
emg_coeffs = pywt.wavedec(
    n_emg, "sym5", level=max_levels, axis=-1, mode="periodization"
)
mix_coeffs = pywt.wavedec(mix, "sym5", level=max_levels, axis=-1, mode="periodization")
fix_coeffs = pywt.wavedec(
    fixed, "sym5", level=max_levels, axis=-1, mode="periodization"
)

level_names = [f"Level $c_{n}$" for n in range(len(eeg_coeffs), 0, -1)]

import scipy.stats

# fig, axes = plt.subplots(nrows=len(eeg_coeffs), figsize=(15, 3.5 * len(eeg_coeffs)), sharex=True)
#
# for cs_eeg, cs_eog, cs_emg, cs_mix, cs_fix, name, ax in zip(eeg_coeffs, eog_coeffs, emg_coeffs, mix_coeffs, fix_coeffs, level_names, axes):
#     ax.hist(cs_eeg.reshape(-1), range=(-10, 10), bins=200, density=True, alpha=0.7, label="EEG")
#     ax.hist(cs_eog.reshape(-1), range=(-10, 10), bins=200, density=True, alpha=0.7, label="EOG")
#     ax.hist(cs_emg.reshape(-1), range=(-10, 10), bins=200, density=True, alpha=0.7, label="EMG")
#     # ax.hist(cs_mix.reshape(-1), range=(-10, 10), bins=200, density=True, alpha=0.7, label="Sum")
#     # ax.hist(cs_fix.reshape(-1), range=(-10, 10), bins=200, density=True, alpha=0.7, label="Fixed")
#
#     ax.set_ylabel(name)
#
# ax.legend()
# ax.set_xlim(-5, 5)
# fig.suptitle("EEG", y=0.90)
# fig
# fig.savefig("figures/theory/wavelet_coeffs_distribution_by_type.svg")

# %%



# %%

FIG_WIDTH = 7.15
FIG_HEIGHT = FIG_WIDTH * 1 / 3

fig, axes = plt.subplots(
    nrows=len(eeg_coeffs),
    sharex=True,
    ncols=3,
    figsize=(FIG_WIDTH, FIG_HEIGHT * len(eeg_coeffs)),
)

n = 0
xlim = (-8, 8)
for cs_eeg, cs_eog, cs_emg, name in zip(
    eeg_coeffs, eog_coeffs, emg_coeffs, level_names
):
    _, bins, _ = axes[n, 0].hist(cs_eeg.reshape(-1), bins=100, density=True, alpha=0.7)
    axes[n, 0].plot(
        bins,
        stats.norm(scale=cs_eeg.std()).pdf(bins),
        ls="--",
        lw=2,
        label=f"σ = {cs_eeg.std():.2f}",
    )
    μ, σ, β = fit_generalized_gaussian(cs_eeg.reshape(-1))
    axes[n, 0].plot(
        bins,
        stats.gennorm.pdf(bins, loc=μ, scale=σ, beta=β),
        ls=":",
        lw=2,
        label=f"σ = {σ:.2f}, β = {β:.2f}",
    )
    axes[n, 0].legend()
    axes[n, 1].hist(cs_eog.reshape(-1), bins=100, density=True, alpha=0.7)
    axes[n, 2].hist(cs_emg.reshape(-1), bins=100, density=True, alpha=0.7)
    axes[n, 0].set_ylabel(name)
    axes[n, 0].set_xlim(*xlim)
    axes[n, 1].set_xlim(*xlim)
    axes[n, 2].set_xlim(*xlim)
    n += 1

axes[0, 0].set_title("EEG")
axes[0, 1].set_title("EOG")
axes[0, 2].set_title("EMG")
# fig.savefig("figures/theory/wavelet_distribution_EEG_EOG_EMG.eps")
# fig.savefig("figures/theory/wavelet_distribution_EEG_EOG_EMG.pdf")

# %% Small figure (EUSIPCO)

import pingouin as pg

pg.normality(eeg_coeffs[0][:100].ravel())

stats.skew(eeg_coeffs[0].ravel())
stats.skew(eeg_coeffs[0].ravel())
stats.skew(eeg_coeffs[1].ravel())
stats.skew(eeg_coeffs[2].ravel())

# %%

import scipy.special

FIG_WIDTH = 2.3 * 7.16

num_levels = 4
fig, axes = plt.subplots(
    nrows=num_levels,
    sharex=True,
    ncols=3,
    figsize=(FIG_WIDTH, FIG_WIDTH * 0.3),
)

num_bins = 201
bin_range = (-10, 10)
n = 0
for cs_eeg, cs_eog, cs_emg, name in zip(
    eeg_coeffs[:num_levels], eog_coeffs[:num_levels], emg_coeffs[:num_levels], level_names[:num_levels]
):
    _, bins, _ = axes[n, 0].hist(
        cs_eeg.reshape(-1), fc="#2D9CDB", bins=num_bins, range=bin_range, density=True, alpha=0.7
    )

    μ, α, β = fit_generalized_gaussian(cs_eeg.reshape(-1))
    axes[n, 0].plot(
        bins,
        stats.gennorm.pdf(bins, loc=μ, scale=α, beta=β),
        c="k",
        ls="--",
        lw=1.5,
        label=f"α = {α:.2f}\nβ = {β:.2f}",
    )
    axes[n, 0].legend(loc="upper right", bbox_to_anchor=(1.03, 1.1))

    if n == 0:
        μ, α, β = fit_generalized_gaussian(cs_eog.reshape(-1))
        axes[n, 1].plot(
            bins,
            stats.gennorm.pdf(bins, loc=μ, scale=α, beta=β),
            c="k",
            ls="--",
            lw=1,
            alpha=0.8,
            label=f"α = {α:.2f}\nβ = {β:.2f}",
        )
        axes[n, 1].legend(loc="upper right", bbox_to_anchor=(1.03, 1.1))

    μ, α, β = fit_generalized_gaussian(cs_emg.reshape(-1))
    axes[n, 2].plot(
        bins,
        stats.gennorm.pdf(bins, loc=μ, scale=α, beta=β),
        c="k",
        ls="--",
        lw=1,
        alpha=0.8 if n == 2 else 1,
        label=f"α = {α:.2f}\nβ = {β:.2f}",
    )
    axes[n, 2].legend(loc="upper right", bbox_to_anchor=(1.03, 1.1))


    axes[n, 2].hist(cs_emg.reshape(-1), fc="#2D9CDB", bins=num_bins, range=bin_range, density=True, alpha=0.7)
    axes[n, 1].hist(cs_eog.reshape(-1), fc="#2D9CDB", bins=num_bins, range=bin_range, density=True, alpha=0.7)
    axes[n, 0].set_ylabel(name)
    axes[n, 0].set_yticks([])
    axes[n, 1].set_yticks([])
    axes[n, 2].set_yticks([])

    n += 1

axes[0, 0].set_title("EEG", fontsize=14)
axes[0, 1].set_title("EOG", fontsize=14)
axes[0, 2].set_title("EMG", fontsize=14)

axes[0, 0].set_xticks([-5, 0, 5])
axes[0, 1].set_xticks([-5, 0, 5])
axes[0, 2].set_xticks([-5, 0, 5])
axes[0, 0].set_xlim(*xlim)
axes[0, 1].set_xlim(*xlim)
axes[0, 2].set_xlim(*xlim)

axes[-1, 0].set_xlabel("Coefficient value", labelpad=-2)
axes[-1, 1].set_xlabel("Coefficient value", labelpad=-2)
axes[-1, 2].set_xlabel("Coefficient value", labelpad=-2)

fig.subplots_adjust(wspace=0.04, hspace=0.1)
# fig.tight_layout()

# fig.savefig("/home/matteo/Research/Papers/2022_Wavelet_CDF/wavelet_coeffs_dist_eusipco.svg")
# fig.savefig("/home/matteo/Research/Papers/2022_Wavelet_CDF/wavelet_coeffs_dist_eusipco.pdf")
fig.savefig("/home/matteo/Research/Papers/2022_Wavelet_CDF/manuscript_eusipco/figures/histograms_v2.pdf")


# %%

cs = eeg_coeffs[0].reshape(-1)
_, bins, _ = plt.hist(cs, bins=50, density=True)
plt.plot(bins, stats.norm(scale=cs.std()).pdf(bins), ls="--", lw=2)
plt.xlim(-10, 10)

cs = eeg_coeffs[1].reshape(-1)
_, bins, _ = plt.hist(cs, bins=100, density=True)
plt.plot(bins, stats.norm(scale=cs.std()).pdf(bins))


cs = eeg_coeffs[2].reshape(-1)
_, bins, _ = plt.hist(cs, bins=100, density=True)
plt.plot(bins, stats.norm(scale=cs.std()).pdf(bins))

cs = eeg_coeffs[3].reshape(-1)
_, bins, _ = plt.hist(cs, bins=100, density=True)
plt.plot(bins, stats.norm(scale=cs.std()).pdf(bins))
plt.xlim(-10, 10)

cs = eeg_coeffs[4].reshape(-1)
_, bins, _ = plt.hist(cs, bins=100, density=True)
plt.plot(bins, stats.norm(scale=cs.std()).pdf(bins))
plt.xlim(-10, 10)

cs = eeg_coeffs[5].reshape(-1)
_, bins, _ = plt.hist(cs, bins=100, density=True)
plt.plot(bins, stats.norm(scale=cs.std()).pdf(bins))
plt.xlim(-10, 10)

# %%

signal = eeg[467]
signal_artifacted = signal + 0.5 * emg[245]

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

plt.savefig("figures/theory/example_signal_artifact_removal.svg")

# %%

num_levels = len(cs_signal) - 1
level_labels = [f"$d_n$" for n in range(num_levels + 1, 1, -1)]

for cs_ref, cs_art, cs_rec, level_label in zip(
    cs_signal, cs_artifact, cs_reconstruction, level_labels
):
    cs_ref_abs = np.abs(cs_ref)
    cs_art_abs = np.abs(cs_art)
    cs_rec_abs = np.abs(cs_rec)

    p = np.linspace(0, 1, cs_ref_abs.size)

    fig, ax = plt.subplots()
    ax.plot(sorted(cs_ref_abs), p, label="Reference signal")
    ax.plot(sorted(cs_art_abs), p, label="Artifacted")
    ax.plot(sorted(cs_rec_abs), p, label="Reconstruction")
    ax.set_title(level_label)
    ax.legend()
    fig.savefig(f"figures/theory/cumulative_coefficient_mapping_{level_label}.svg")


# %%

# import scipy.signal as ss

# freq, S_eeg = ss.welch(n_eeg, fs=256, nperseg=512, noverlap=128)
# freq, S_eog = ss.welch(n_eog, fs=256, nperseg=512, noverlap=128)
# freq, S_emg = ss.welch(n_emg, fs=256, nperseg=512, noverlap=128)

# fig, ax = plt.subplots()
# ax.plot(freq, S_eeg.mean(axis=0), label="EEG")
# ax.plot(freq, S_eog.mean(axis=0), label="EOG")
# ax.plot(freq, S_emg.mean(axis=0), label="EMG")
# ax.set_xlim(0, 40)
# ax.set_xlabel("Frequency (Hz)")
# ax.legend()

# fig.savefig("figures/theory/psd_by_type.svg")

# %%

# %%
