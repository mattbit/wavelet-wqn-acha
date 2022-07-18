import pywt
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

plt.style.use("resources/figstyle.mplstyle")

FIG_WIDTH = 2.3 * 7.16


# Gaussian fitting utils
from scipy import optimize


def fit_generalized_gaussian(x):
    μ0 = x.mean()
    σ0 = x.std()
    β0 = 2

    res = optimize.minimize(
        neg_gg_likelihood,
        (μ0, σ0, β0),
        args=(x,),
        bounds=[(-np.inf, np.inf), (1e-2, np.inf), (1e-2, np.inf)],
    )

    return res.x


def neg_gg_likelihood(θ, x):
    μ, σ, β = θ
    return -stats.gennorm.logpdf(x, loc=μ, scale=σ, beta=β).sum()


# Load data and normalize
eeg = np.load("resources/data/eeg-denoise-net/EEG_all_epochs.npy")
eog = np.load("resources/data/eeg-denoise-net/EOG_all_epochs.npy")
emg = np.load("resources/data/eeg-denoise-net/EMG_all_epochs.npy")

n_eeg = (eeg - eeg.mean(axis=-1).reshape(-1, 1)) / eeg.std(axis=-1).reshape(-1, 1)
n_eog = (eog - eog.mean(axis=-1).reshape(-1, 1)) / eog.std(axis=-1).reshape(-1, 1)
n_emg = (emg - emg.mean(axis=-1).reshape(-1, 1)) / emg.std(axis=-1).reshape(-1, 1)


# Wavelet transform
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

level_names = [f"Level $c_{n}$" for n in range(len(eeg_coeffs), 0, -1)]


# Prepare figure

num_levels = 4
xlim = (-8, 8)

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
    eeg_coeffs[:num_levels],
    eog_coeffs[:num_levels],
    emg_coeffs[:num_levels],
    level_names[:num_levels],
):
    _, bins, _ = axes[n, 0].hist(
        cs_eeg.reshape(-1),
        fc="#2D9CDB",
        bins=num_bins,
        range=bin_range,
        density=True,
        alpha=0.7,
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

    axes[n, 2].hist(
        cs_emg.reshape(-1),
        fc="#2D9CDB",
        bins=num_bins,
        range=bin_range,
        density=True,
        alpha=0.7,
    )
    axes[n, 1].hist(
        cs_eog.reshape(-1),
        fc="#2D9CDB",
        bins=num_bins,
        range=bin_range,
        density=True,
        alpha=0.7,
    )
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

fig.savefig("./output/acha_fig2_coeff_distributions.pdf")
print("Figure saved to `./output/acha_fig2_coeff_distributions.pdf`")
