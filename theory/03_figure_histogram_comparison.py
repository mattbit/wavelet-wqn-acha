# %%

import pywt
import numpy as np
from fbm import FBM
import scipy.signal as ss
import matplotlib.pyplot as plt

from methods import WQNDenoiser, WTDenoiser, mse

FIG_WIDTH = 2 * 7.16


COLORS = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]

COLOR_ORIG = COLORS[6]
COLOR_WT_HARD = COLORS[7]
COLOR_WT_SOFT = COLORS[3]
COLOR_WQN = COLORS[5]
COLOR_UNI_SOFT = COLORS[6]


plt.style.use("hpub")
plt.rcParams["figure.figsize"] = (FIG_WIDTH, FIG_WIDTH)

# %%

np.random.seed(3884)

hurst = 0.5
t = np.linspace(0, 3, 2**13)
s1 = FBM(n=10_000, hurst=hurst, length=1).fbm()[: len(t)]
s2 = FBM(n=10_000, hurst=hurst, length=1).fbm()[: len(t)]

a1 = ss.square(100 * t / np.pi)
a2 = ss.sawtooth(100 * t / np.pi, 0.5)

x1 = s1 + a1
x2 = s2 + a2

# %%

denoiser_wqn = WQNDenoiser(wavelet="sym3", mode="symmetric")
denoiser_wt_isoft = WTDenoiser(
    method="ideal_soft", wavelet="sym3", mode="symmetric"
)
denoiser_wt_ihard = WTDenoiser(
    method="ideal_hard", wavelet="sym3", mode="symmetric"
)

# %%

x_wt_isoft1, cs_wt_isoft1 = denoiser_wt_isoft.denoise(x1, s1, with_coeffs=True)
x_wt_ihard1, cs_wt_ihard1 = denoiser_wt_ihard.denoise(x1, s1, with_coeffs=True)
x_wqn1, cs_wqn1 = denoiser_wqn.denoise(x1, s1, with_coeffs=True)

x_wt_isoft2, cs_wt_isoft2 = denoiser_wt_isoft.denoise(x2, s2, with_coeffs=True)
x_wt_ihard2, cs_wt_ihard2 = denoiser_wt_ihard.denoise(x2, s2, with_coeffs=True)
x_wqn2, cs_wqn2 = denoiser_wqn.denoise(x2, s2, with_coeffs=True)

cs_ref1 = pywt.wavedec(s1, denoiser_wqn.wavelet, mode=denoiser_wqn.mode)
cs_ref2 = pywt.wavedec(s2, denoiser_wqn.wavelet, mode=denoiser_wqn.mode)


print("MSE WQN", mse(s1, x_wqn1))
print("MSE WT hard", mse(s1, x_wt_ihard1))
print("MSE WT soft", mse(s1, x_wt_isoft1))

# %%
# Prepare the figure

plot_level = 6
spacing = 1.4
num_bins = 60

_range = (-0.3, 0.3)

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, 2 * 0.3 * FIG_WIDTH),
    ncols=2,
    nrows=2,
)

# Square artifact
ax[0, 0].plot(t, s1)
ax[0, 0].plot(t, -1 * spacing + x1, c=COLOR_ORIG)

ax[0, 0].plot(t, -2 * spacing + x_wt_ihard1, c=COLOR_WT_HARD)
ax[0, 0].plot(t, -3 * spacing + x_wt_isoft1, c=COLOR_WT_SOFT)
ax[0, 0].plot(t, -4 * spacing + x_wqn1, c=COLOR_WQN)

ax[0, 0].set_yticks(-np.arange(5) * spacing)
ax[0, 0].set_yticklabels(["Test signal", "Artifacted", "HT", "ST", "WQN"])

ax[0, 1].hist(
    cs_ref1[plot_level],
    bins=num_bins,
    range=_range,
    density=True,
    fc="black",
    label="Test signal",
)

ax[0, 1].hist(
    cs_wt_ihard1[plot_level],
    bins=num_bins,
    range=_range,
    density=True,
    fc=COLOR_WT_HARD,
    alpha=0.75,
    label="Hard thresholding (HT)",
)
ax[0, 1].hist(
    cs_wt_isoft1[plot_level],
    bins=num_bins,
    range=_range,
    density=True,
    alpha=0.75,
    fc=COLOR_WT_SOFT,
    label="Soft thresholding (ST)",
)
ax[0, 1].hist(
    cs_wqn1[plot_level],
    bins=num_bins,
    range=_range,
    density=True,
    fc=COLOR_WQN,
    alpha=0.75,
    label="WQN",
)
ax[0, 1].legend()

# Triangle artefact
ax[1, 0].plot(t, s2)
ax[1, 0].plot(t, -1 * spacing + x2, c=COLOR_ORIG)

ax[1, 0].plot(t, -2 * spacing + x_wt_ihard2, c=COLOR_WT_HARD)
ax[1, 0].plot(t, -3 * spacing + x_wt_isoft2, c=COLOR_WT_SOFT)
ax[1, 0].plot(t, -4 * spacing + x_wqn2, c=COLOR_WQN)

ax[1, 0].set_yticks(-np.arange(5) * spacing)
ax[1, 0].set_yticklabels(["Test signal", "Artifacted", "HT", "ST", "WQN"])

ax[1, 1].hist(
    cs_ref2[plot_level],
    bins=num_bins,
    range=_range,
    density=True,
    fc="black",
    label="Test signal",
)

ax[1, 1].hist(
    cs_wt_ihard2[plot_level],
    bins=num_bins,
    range=_range,
    density=True,
    fc=COLOR_WT_HARD,
    alpha=0.75,
    label="Hard thresholding (HT)",
)
ax[1, 1].hist(
    cs_wt_isoft2[plot_level],
    bins=num_bins,
    range=_range,
    density=True,
    alpha=0.75,
    fc=COLOR_WT_SOFT,
    label="Soft thresholding (ST)",
)
ax[1, 1].hist(
    cs_wqn2[plot_level],
    bins=num_bins,
    range=_range,
    density=True,
    fc=COLOR_WQN,
    alpha=0.75,
    label="WQN",
)

ax[0, 0].set_title("A1. Square wave artifact", loc="left")
ax[0, 1].set_title("A2. Distribution of wavelet coefficients (square wave)", loc="left")

ax[1, 0].set_title("B1. Triangle wave artifact", loc="left")
ax[1, 1].set_title(
    "B2. Distribution of wavelet coefficients (triangle wave)", loc="left"
)

ax[0, 0].set_xlabel("Time")
ax[1, 0].set_xlabel("Time")
ax[0, 1].set_xlabel("Coefficient value")
ax[1, 1].set_xlabel("Coefficient value")
ax[0, 0].set_xticks([])
ax[1, 0].set_xticks([])
ax[0, 1].yaxis.set_visible(False)
ax[1, 1].yaxis.set_visible(False)

ax[0, 0].set_xlim(0, 2)
ax[1, 0].set_xlim(0, 2)

ax[0, 1].set_xlim(*_range)
ax[1, 1].set_xlim(*_range)


fig.tight_layout()

fig.savefig(f"acha_fig3_hurst{hurst}_v2.pdf")


# %%
