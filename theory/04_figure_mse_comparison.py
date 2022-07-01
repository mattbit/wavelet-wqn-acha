# %%

import pywt
import numpy as np
import pandas as pd
from fbm import FBM
from tqdm import trange
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

hurst = 0.5

denoiser_wqn = WQNDenoiser(wavelet="sym3", mode="periodization")
denoiser_wt_isoft = WTDenoiser(
    method="ideal_soft", wavelet="sym3", mode="periodization"
)
denoiser_wt_ihard = WTDenoiser(
    method="ideal_hard", wavelet="sym3", mode="periodization"
)
denoiser_wt_usoft = WTDenoiser(
    method="universal_soft", wavelet="sym3", mode="periodization"
)

denoisers = [
    ("wqn", denoiser_wqn),
    ("wt_ihard", denoiser_wt_ihard),
    ("wt_isoft", denoiser_wt_isoft),
    ("wt_usoft", denoiser_wt_usoft),
]

denoisers_attrs = {
    "wqn": {"color": COLOR_WQN, "marker": "o", "label": "WQN"},
    "wt_isoft": {
        "color": COLOR_WT_SOFT,
        "marker": "D",
        "label": "Ideal soft thresholding",
    },
    "wt_ihard": {
        "color": COLOR_WT_HARD,
        "marker": "s",
        "label": "Ideal hard thresholding",
    },
    "wt_usoft": {
        "color": COLOR_UNI_SOFT,
        "marker": "v",
        "label": "Universal thresholding",
    },
}

# %%
# Compute statistics

np.random.seed(3884)

scales = np.arange(0, 21)
num_realizations = 100
t = np.linspace(0, 3, 2**12)

# Data array (realization, square/triangle, time)
data = {dname: np.zeros((num_realizations, 2, t.size)) for dname, _ in denoisers}

mse_slice = slice(t.size // 4, 3 * t.size // 4)
_records = []
for n in trange(num_realizations):
    # Square wave
    s = FBM(n=len(t), hurst=hurst, length=1).fbm()[: len(t)]
    s /= s.std()

    a = ss.square((100 * t + np.random.randint(0, 99)) / np.pi)
    a /= a.std()

    for scale in scales:
        x = s + scale * a

        for name, denoiser in denoisers:
            _records.append(
                {
                    "denoiser": name,
                    "realization": n,
                    "scale": scale,
                    "mse": mse(denoiser.denoise(x, s)[mse_slice], s[mse_slice]),
                }
            )

pd.DataFrame(_records).to_csv(f"data_mse_comparison_hurst{hurst}.csv", index=False)

# %% Plot figure.

fig, (ax0, ax1) = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH / 2), ncols=2)

# Figure A

df = pd.read_csv(f"data_mse_comparison_hurst{hurst}.csv")
dx = (
    df.loc[:, ("denoiser", "scale", "mse")]
    .groupby(["denoiser", "scale"])
    .agg(["mean", "sem", "std", "median"])
)

lowq = df.groupby(["denoiser", "scale"]).mse.quantile(0.25)
highq = df.groupby(["denoiser", "scale"]).mse.quantile(1 - 0.25)

for name, _ in denoisers:
    (l,) = ax0.plot(
        scales,
        dx.loc[name, ("mse", "median")],
        marker=denoisers_attrs[name]["marker"],
        markersize=8,
        lw=1.5,
        c=denoisers_attrs[name]["color"],
        label=denoisers_attrs[name]["label"],
    )
    ax0.fill_between(
        scales, lowq.loc[name], highq.loc[name], fc=l.get_color(), alpha=0.2
    )

ax0.legend()
ax0.set_ylabel("MSE after artifact removal")
ax0.set_xlabel("Artifact amplitude (standard deviation)")
ax0.set_xticks(scales[::2])

ax0.set_title("A. Comparison of reconstruction error", loc="left")

# Variances
np.random.seed(38988)

t = np.linspace(0, 3, 2**14)

s1 = FBM(n=t.size, hurst=hurst, length=1).fbm()[: len(t)]
s1 /= s1.std()

a1 = ss.square((100 * t + np.random.randint(0, 99)) / np.pi)
a1 /= a1.std()

x1 = s1 + 2 * a1

x_fig, x_ax = plt.subplots()
x_ax.plot(s1)
i = 0
for name, denoiser in denoisers[:3]:
    i += 1
    x_, cs_ = denoiser.denoise(x1, s1, with_coeffs=True)
    x_ax.plot(x_ - 2 * i, label=denoisers_attrs[name]["label"])
    fs, Pxx = ss.welch(x_, nperseg=1024)
    ax1.plot(
        fs,
        Pxx,
        marker=denoisers_attrs[name]["marker"],
        lw=1.5,
        c=denoisers_attrs[name]["color"],
        label=denoisers_attrs[name]["label"] + f" (MSE = {mse(x_, s1).round(2)})",
    )


fs, Pxx = ss.welch(s1, nperseg=1024)
ax1.plot(
    fs,
    Pxx,
    label="Non-artifacted signal",
    c=COLORS[0],
    ls="--",
    lw=1.5,
)

ax1.legend()

ax1.set_yscale("log")
ax1.set_xscale("log")

ax1.set_xlabel("Frequency")
ax1.set_ylabel("Power spectral density")

ax1.set_title("B. Spectral distortion", loc="left")

fig.savefig("acha_fig3b_quantitative_comparison.pdf")


# %% Variances

np.random.seed(38989)

t = np.linspace(0, 3, 2**14)

s1 = FBM(n=t.size, hurst=hurst, length=1).fbm()[: len(t)]
s1 /= s1.std()

a1 = ss.square((100 * t + np.random.randint(0, 99)) / np.pi)
a1 /= a1.std()

x1 = s1 + 2 * a1

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH / 2))

σ = np.diff(s1).std()

cs_ref = denoiser_wqn._dwt(s1)
jj = np.arange(len(cs_ref), 1, -1)

ax.plot(2**jj, [c.var() for c in cs_ref[1:]])

for name, denoiser in denoisers[:3]:
    i += 1
    x_, cs_ = denoiser.denoise(x1, s1, with_coeffs=True)
    
    fs, Pxx = ss.welch(x_, nperseg=1024)
    ax.plot(
        2**jj,
        [c.var() for c in cs_[1:]],
        marker=denoisers_attrs[name]["marker"],
        lw=1.5,
        c=denoisers_attrs[name]["color"],
        label=denoisers_attrs[name]["label"] + f" (MSE = {mse(x_, s1).round(2)})",
    )

ax.plot(2**jj[4:9], 0.1 * σ**2 / 2 * 2**(2 * jj[4:9]), ls="--", c="gray", label="Theoretical slope $\mathrm{Var}[d_j] \sim 2^{2j}$")
# ax.plot(2**jj, σ**2 / 2 * ((2**jj)**(2 * hurst + 1)), ls="--", c="gray", label="Theoretical slope ($(2^j)^2$")

ax.set_yscale("log")
ax.set_xscale("log", base=2)

ax.set_ylabel("Variance of wavelet coefficients $d_j$")
ax.set_xlabel("Scale $2^j$")

ax.legend()

fig.savefig("acha_fig3c_scaling_behaviour.pdf")

# %%




