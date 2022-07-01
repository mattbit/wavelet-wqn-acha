# %%

import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from fbm import FBM
from methods import WQNDenoiser, WTDenoiser

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

t = np.linspace(0, 3, 10000)
s1 = FBM(n=10_000, hurst=hurst, length=1).fbm()[: len(t)]
s2 = FBM(n=10_000, hurst=hurst, length=1).fbm()[: len(t)]

a1 = ss.square(100 * t / np.pi)
a2 = ss.sawtooth(100 * t / np.pi, 0.5)

x1 = s1 + a1
x2 = s2 + a2

# %%

denoiser_wqn = WQNDenoiser()
denoiser_wt_ihard = WTDenoiser(method="ideal_soft")
denoiser_wt_isoft = WTDenoiser(method="ideal_hard")

# %%

x_wt_isoft1, cs_wt_isoft1 = denoiser_wt_isoft.denoise(x1, s1, with_coeffs=True)
x_wt_ihard1, cs_wt_ihard1 = denoiser_wt_ihard.denoise(x1, s1, with_coeffs=True)
x_wqn1, cs_wqn1 = denoiser_wqn.denoise(x1, s1, with_coeffs=True)

x_wt_isoft2, cs_wt_isoft2 = denoiser_wt_isoft.denoise(s2, x2, with_coeffs=True)
x_wt_ihard2, cs_wt_ihard2 = denoiser_wt_ihard.denoise(s2, x2, with_coeffs=True)
x_wqn2, cs_wqn2 = denoiser_wqn.denoise(x2, s2, with_coeffs=True)

cs_ref1 = pywt.wavedec(s1, denoiser_wqn.wavelet, mode=denoiser_wqn.mode)
cs_ref2 = pywt.wavedec(s2, denoiser_wqn.wavelet, mode=denoiser_wqn.mode)

# %%
# Prepare the figure

plot_level = 4
spacing = 1.4
num_bins = 60

_range = (
    -1.1 * np.max([cs_wt_isoft1[plot_level], cs_wt_isoft2[plot_level]]),
    1.1 * np.max([cs_wt_isoft1[plot_level], cs_wt_isoft2[plot_level]]),
)

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

fig.savefig("acha_fig3_hurst{hurst}_v2.pdf")

# %%

# Compute statistics

from tqdm import trange

np.random.seed(3884)

num_realizations = 1_000

t = np.linspace(0, 3, 10_000)

mse1 = []
mse2 = []


def _mse(x, y):
    return np.mean((x - y) ** 2)


for n in trange(num_realizations):
    s1 = FBM(n=10_000, hurst=hurst, length=1).fbm()[: len(t)]
    s1 /= s1.std()
    s2 = FBM(n=10_000, hurst=hurst, length=1).fbm()[: len(t)]
    s2 /= s2.std()

    a1 = ss.square((100 * t + np.random.randint(0, 99)) / np.pi)
    a1 /= a1.std()
    a2 = ss.sawtooth((100 * t + np.random.randint(0, 99)) / np.pi, 0.5)
    a2 /= a2.std()

    x1 = s1 + a1
    x2 = s2 + a2

    x_wt_isoft1 = remove_artifact_wt(s1, x1, "ideal_soft", num_levels)
    x_wt_ihard1 = remove_artifact_wt(s1, x1, "ideal_hard", num_levels)
    x_wt_uni_soft1 = remove_artifact_wt(s1, x1, "universal_soft", num_levels)
    x_wt_uni_hard1 = remove_artifact_wt(s1, x1, "universal_hard", num_levels)
    # x_wt_sure1 = remove_artifact_wt(s1, x1, "sure", num_levels)
    x_wqn1 = remove_artifact(s1, x1, num_levels)

    x_wt_soft2 = remove_artifact_wt(s2, x2, "ideal_soft", num_levels)
    x_wt_ihard2 = remove_artifact_wt(s2, x2, "ideal_hard", num_levels)
    x_wt_uni_soft2 = remove_artifact_wt(s2, x2, "universal_soft", num_levels)
    x_wt_uni_hard2 = remove_artifact_wt(s2, x2, "universal_hard", num_levels)
    # x_wt_sure2 = remove_artifact_wt(s2, x2, "sure", num_levels)
    x_wqn2 = remove_artifact(s2, x2, num_levels)

    mse1.append(
        {
            "wt_soft": _mse(x_wt_isoft1, s1),
            "wt_hard": _mse(x_wt_ihard1, s1),
            "wt_uni_soft": _mse(x_wt_uni_soft1, s1),
            "wt_uni_hard": _mse(x_wt_uni_hard1, s1),
            # "wt_sure": _mse(x_wt_sure1, s1),
            "wqn": _mse(x_wqn1, s1),
        }
    )

    mse2.append(
        {
            "wt_soft": _mse(x_wt_isoft2, s2),
            "wt_hard": _mse(x_wt_ihard2, s2),
            "wt_uni_soft": _mse(x_wt_uni_soft2, s2),
            "wt_uni_hard": _mse(x_wt_uni_hard2, s2),
            # "wt_sure": _mse(x_wt_sure2, s2),
            "wqn": _mse(x_wqn2, s2),
        }
    )


# %%

import pandas as pd

df1 = pd.DataFrame(mse1)
df2 = pd.DataFrame(mse2)


df1.to_csv("df1_comp.csv", index=False)
df2.to_csv("df2_comp.csv", index=False)


# %%


df1 = pd.read_csv("df1_comp.csv")
df2 = pd.read_csv("df2_comp.csv")

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH / 2))
ax.hist(
    df1.wt_soft,
    bins=75,
    range=(0, 1.0),
    alpha=0.75,
    fc=COLOR_WT_SOFT,
    label="Ideal ST",
    density=True,
)
ax.hist(
    df1.wqn,
    bins=75,
    range=(0, 1.0),
    alpha=0.75,
    fc=COLOR_WQN,
    label="WQN",
    density=True,
)
ax.hist(
    df1.wt_uni_soft,
    bins=75,
    range=(0, 1.0),
    alpha=0.75,
    fc=COLOR_ORIG,
    label="Universal ST",
    density=True,
)
ax.set_ylabel("Density")
ax.set_xlabel("Mean Squared Error")
ax.legend()

fig.savefig("e_acha_mse.svg")

# %%

plt.hist(df2.wqn, bins=150, range=(0, 1.0), alpha=0.5, label="WQN")
plt.hist(df2.wt_soft, bins=150, range=(0, 1.0), alpha=0.5, label="Ideal ST")
plt.hist(df2.wt_hard, bins=150, range=(0, 1.0), alpha=0.5, label="Ideal HT")
plt.hist(df2.wt_uni_soft, bins=150, range=(0, 1.0), alpha=0.5, label="Universal ST")
plt.legend()

# %%

import scipy.stats

pp = np.linspace(0, 1, 1000)

plt.plot(pp, scipy.stats.gaussian_kde(df1.wqn)(pp), label="WQN")
plt.plot(pp, scipy.stats.gaussian_kde(df1.wt_soft)(pp), label="Ideal ST")
plt.plot(pp, scipy.stats.gaussian_kde(df1.wt_hard)(pp), label="Ideal HT")
plt.plot(pp, scipy.stats.gaussian_kde(df1.wt_uni_soft)(pp), label="Universal ST")
plt.legend()

# %%


df1.mean().round(2)
df2.mean().round(2)

df1.quantile(0.25).round(2)
df1.quantile(0.75).round(2)


# %%

vmse = []

# %%

scales = np.arange(1, 21)

for n in trange(num_realizations):
    s1 = FBM(n=10_000, hurst=hurst, length=1).fbm()[: len(t)]
    s1 /= s1.std()

    a1 = ss.square((100 * t + np.random.randint(0, 99)) / np.pi)
    a1 /= a1.std()

    for scale in scales:
        x1 = s1 + scale * a1

        x_wt_isoft = remove_artifact_wt(
            s1, x1, "ideal_soft", level=None, mode="periodization", wavelet="sym3"
        )
        x_wt_ihard = remove_artifact_wt(
            s1, x1, "ideal_hard", level=None, mode="periodization", wavelet="sym3"
        )
        x_wt_uni_soft = remove_artifact_wt(
            s1, x1, "universal_soft", level=None, mode="periodization", wavelet="sym3"
        )
        x_wt_uni_hard = remove_artifact_wt(
            s1, x1, "universal_hard", level=None, mode="periodization", wavelet="sym3"
        )
        x_wqn = remove_artifact(s1, x1, None, mode="periodization", wavelet="sym3")

        vmse.append(
            {
                "wt_soft": _mse(x_wt_isoft, s1),
                "wt_hard": _mse(x_wt_ihard, s1),
                "wt_uni_soft": _mse(x_wt_uni_soft, s1),
                "wt_uni_hard": _mse(x_wt_uni_hard, s1),
                # "wt_sure": _mse(x_wt_sure1, s1),
                "wqn": _mse(x_wqn, s1),
                "scale": scale,
            }
        )

# %%

df = pd.DataFrame(vmse)
df.to_csv("df_comp_scales.csv", index=False)

# %%

df = pd.read_csv("df_comp_scales.csv")
dfv = df.groupby("scale").agg(["mean", "sem", "std"])

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH / 2))

lowq = df.groupby("scale").quantile(0.25)
highq = df.groupby("scale").quantile(1 - 0.25)

items = [
    ("wqn", COLOR_WQN, "o", "WQN"),
    ("wt_soft", COLOR_WT_SOFT, "D", "Ideal soft thresholding"),
    ("wt_hard", COLOR_WT_HARD, "s", "Ideal hard thresholding"),
    ("wt_uni_soft", COLOR_UNI_SOFT, "v", "Universal thresholding"),
]

for item, clr, mkr, lbl in items:
    (l,) = ax.plot(scales, dfv.loc[:, (item, "mean")], marker=mkr, markersize=8, lw=1.5, c=clr, label=lbl)
    # ax.errorbar(scales, dfv.loc[:, (item, "mean")], dfv.loc[:, (item, "std")], lw=2)
    ax.fill_between(scales, lowq[item], highq[item], fc=l.get_color(), alpha=0.1)

ax.legend()
ax.set_ylabel("MSE after artifact removal")
ax.set_xlabel("Artifact amplitude (standard deviation)")
ax.set_xticks(scales[1::2])

# %%

x_wt_isoft = remove_artifact_wt(s1, x1, "ideal_soft", None, mode="periodization")
x_wqn = remove_artifact(s1, x1, None, mode="periodization")

_mse(
    remove_artifact_wt(s1, x1, "ideal_soft", None, wavelet="haar", mode="periodic"), s1
)
_mse(
    remove_artifact_wt(
        s1, x1, "ideal_soft", None, wavelet="haar", mode="periodization"
    ),
    s1,
)
_mse(
    remove_artifact_wt(s1, x1, "ideal_soft", None, wavelet="haar", mode="symmetric"), s1
)
_mse(
    remove_artifact_wt(
        s1, x1, "ideal_soft", None, wavelet="haar", mode="antisymmetric"
    ),
    s1,
)
_mse(
    remove_artifact_wt(s1, x1, "ideal_soft", None, wavelet="haar", mode="constant"), s1
)
_mse(remove_artifact_wt(s1, x1, "ideal_soft", None, wavelet="haar", mode="reflect"), s1)
_mse(
    remove_artifact_wt(s1, x1, "ideal_soft", None, wavelet="haar", mode="antireflect"),
    s1,
)


_mse(remove_artifact(s1, x1, None, wavelet="sym2", mode="periodic"), s1)
_mse(remove_artifact(s1, x1, None, wavelet="sym2", mode="periodization"), s1)
_mse(remove_artifact(s1, x1, None, wavelet="sym2", mode="symmetric"), s1)
_mse(remove_artifact(s1, x1, None, wavelet="sym2", mode="antisymmetric"), s1)
_mse(remove_artifact(s1, x1, None, wavelet="sym2", mode="constant"), s1)
_mse(remove_artifact(s1, x1, None, wavelet="sym2", mode="reflect"), s1)
_mse(remove_artifact(s1, x1, None, wavelet="sym2", mode="antireflect"), s1)


# %%

plt.plot(s1)
plt.plot(
    remove_artifact_wt(s1, x1, "ideal_soft", None, wavelet="haar", mode="periodization")
)
plt.plot(remove_artifact(s1, x1, None, wavelet="haar", mode="periodization"))

_mse(
    remove_artifact_wt(
        s1, x1, "ideal_soft", None, wavelet="haar", mode="periodization"
    ),
    s1,
)
_mse(remove_artifact(s1, x1, None, wavelet="haar", mode="periodization"), s1)

_mse(
    remove_artifact_wt(
        s1, x1, "ideal_soft", None, wavelet="sym3", mode="periodization"
    ),
    s1,
)
_mse(remove_artifact(s1, x1, None, wavelet="sym3", mode="periodization"), s1)


# %%

# [_mse(remove_artifact_wt(s1, x1, "ideal_soft", l, mode="periodization"), s1) for l in range(1, 14)]
# [_mse(remove_artifact(s1, x1, l, mode="periodization"), s1) for l in range(1, 14)]

# plt.plot(x_wt_isoft)
plt.plot(x_wqn)
plt.plot(remove_artifact_wt(s1, x1, "ideal_soft", 12, mode="periodization"))
plt.plot(s1)


# %% Variances

s1 = FBM(n=10_000, hurst=hurst, length=1).fbm()[: len(t)]
s1 /= s1.std()

a1 = ss.square((100 * t + np.random.randint(0, 99)) / np.pi)
a1 /= a1.std()

x1 = s1 + a1

x_wqn = remove_artifact(s1, x1, mode="periodization", wavelet="sym3")
x_wt_isoft = remove_artifact_wt(s1, x1, mode="periodization", wavelet="sym3")

print("WQN", _mse(x_wqn, s1))
print("WT", _mse(x_wt_isoft, s1))

# %%

s_coeffs = pywt.wavedec(s1, "sym3", mode="periodization")

wqn_coeffs = pywt.wavedec(x_wqn, "sym3", mode="periodization")
wt_coeffs = pywt.wavedec(x_wt_isoft, "sym3", mode="periodization")

plt.plot([cs.std() for cs in s_coeffs], marker="o")
plt.plot([cs.std() for cs in wqn_coeffs], marker="o")
plt.plot([cs.std() for cs in wt_coeffs], marker="o")
plt.yscale("log")

# %%

plt.plot(s1)
plt.plot(x_wqn - 2)
plt.plot(x_wt_isoft - 4)

# %%
