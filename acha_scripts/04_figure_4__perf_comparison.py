import numpy as np
import pandas as pd
from fbm import FBM
from tqdm import trange
from pathlib import Path
import scipy.signal as ss
import matplotlib.pyplot as plt

from methods import WQNDenoiser, WTDenoiser, ZeroDenoiser, mse

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


plt.style.use("resources/figstyle.mplstyle")
plt.rcParams["figure.figsize"] = (FIG_WIDTH, FIG_WIDTH)


# Prepare data and denoisers
a_type = "square"
hurst = 0.5

f_art = getattr(ss, a_type)

denoiser_wqn = WQNDenoiser(wavelet="sym3", mode="periodization")
denoiser_wt_isoft = WTDenoiser(
    method="ideal_soft", wavelet="sym3", mode="periodization"
)
denoiser_wt_ihard = WTDenoiser(
    method="ideal_hard", wavelet="sym3", mode="periodization"
)
denoiser_wt_uni = WTDenoiser(
    method="universal_soft", wavelet="sym3", mode="periodization"
)
denoiser_wt_sure = WTDenoiser(method="sure", wavelet="sym3", mode="periodization")
denoiser_zero = ZeroDenoiser(wavelet="sym3", mode="periodization")

denoisers = [
    ("wqn", denoiser_wqn),
    ("wt_ihard", denoiser_wt_ihard),
    ("wt_isoft", denoiser_wt_isoft),
    ("wt_uni", denoiser_wt_uni),
    ("wt_sure", denoiser_wt_sure),
    ("wt_zero", denoiser_zero),
]

denoisers_attrs = {
    "wqn": {"color": COLOR_WQN, "marker": "o", "label": "WQN"},
    "wt_isoft": {
        "color": COLOR_WT_SOFT,
        "marker": "D",
        "label": "Optimal ST",
    },
    "wt_ihard": {
        "color": COLOR_WT_HARD,
        "marker": "s",
        "label": "Optimal HT",
    },
    "wt_uni": {
        "color": COLOR_UNI_SOFT,
        "marker": "v",
        "label": "Universal thresholding",
    },
    "wt_sure": {
        "color": COLORS[4],
        "marker": "*",
        "label": "SureShrink",
    },
    "wt_zero": {
        "color": "gray",
        "marker": ".",
        "label": "Low-pass",
    },
}


# Compute statistics

np.random.seed(3884)

scales = np.arange(0, 21)
num_realizations = 1_000
t = np.linspace(0, 3, 2**12)

# We compute the MSE just in the central part of the signal to avoid
# considering border effects.
mse_slice = slice(t.size // 4, 3 * t.size // 4)

if not Path(
    f"output/data_mse_comparison_hurst{hurst}_N{num_realizations}_{a_type}.csv"
).exists():
    _records = []
    for n in trange(num_realizations):
        # Square wave
        s = FBM(n=len(t), hurst=hurst, length=1).fbm()[: len(t)]
        s /= s.std()

        a = f_art((100 * t + np.random.randint(0, 99)) / np.pi)
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

    pd.DataFrame(_records).to_csv(
        f"output/data_mse_comparison_hurst{hurst}_N{num_realizations}_{a_type}.csv",
        index=False,
    )


# Plot figure
fig, (ax0, ax1) = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH / 2), ncols=2)

# Figure A: MSE

df = pd.read_csv(f"output/data_mse_comparison_hurst{hurst}_N{num_realizations}_{a_type}.csv")
dx = (
    df.loc[:, ("denoiser", "scale", "mse")]
    .groupby(["denoiser", "scale"])
    .agg(["mean", "sem", "std", "median"])
)

lowq = df.groupby(["denoiser", "scale"]).mse.quantile(0.25)
highq = df.groupby(["denoiser", "scale"]).mse.quantile(1 - 0.25)

for name, _ in denoisers[:-1]:
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
        scales,
        lowq.loc[name],
        highq.loc[name],
        fc=l.get_color(),
        alpha=0.2,
    )

# Low-pass
ax0.plot(
    scales,
    dx.loc["wt_zero", ("mse", "median")],
    lw=1.5,
    ls=":",
    c=denoisers_attrs["wt_zero"]["color"],
    label=denoisers_attrs["wt_zero"]["label"],
)

ax0.legend()
ax0.set_ylabel("MSE after artifact removal")
ax0.set_xlabel("Artifact amplitude (standard deviation)")
ax0.set_xticks(scales[::2])

ax0.set_title("A. Comparison of reconstruction error", loc="left")

# Figure B: Spectrum
np.random.seed(38988)

t = np.linspace(0, 3, 2**14)

s1 = FBM(n=t.size, hurst=hurst, length=1).fbm()[: len(t)]
s1 /= s1.std()

a1 = f_art((100 * t + np.random.randint(0, 99)) / np.pi)
a1 /= a1.std()

x1 = s1 + 2 * a1

x_fig, x_ax = plt.subplots()
x_ax.plot(s1)
i = 0
for name, denoiser in denoisers[:-1]:
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
        label=denoisers_attrs[name]["label"],
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


ax1.plot(
    fs[1:10],
    8e-5 / fs[1:10] ** 2,
    # label="Theoretical slope",
    c=COLORS[0],
    ls=":",
    lw=1.5,
)
ax1.text(
    fs[3],
    5e-5 / fs[3] ** 2 + 8,
    "Theoretical slope (H=½)",
    va="center",
    ha="center",
    rotation=-42,
)

ax1.legend()

ax1.set_yscale("log")
ax1.set_xscale("log")

ax1.set_xlabel("Frequency")
ax1.set_ylabel("Power spectral density")

ax1.set_title("B. Spectral distortion", loc="left")

fig.savefig("./output/acha_fig4_perf_comparison.pdf")
print("Figure saved to `./output/acha_fig4_perf_comparison.pdf`.")
