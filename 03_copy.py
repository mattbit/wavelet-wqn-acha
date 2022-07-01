# %%

import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from fbm import FBM

plt.style.use("hpub")
plt.rcParams["figure.figsize"] = (18, 9)

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

plt.plot([1, 2, 3], c=COLORS[7])

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

num_levels = None
_wavelet = "sym3"
_mode = "symmetric"


def remove_artifact(reference, artifact, level=None, mode="symmetric", wavelet="sym3"):
    cs_signal = pywt.wavedec(reference, wavelet, level=level, mode=mode)
    cs_artifact = pywt.wavedec(artifact, wavelet, level=level, mode=mode)

    coeffs = [c.copy() for c in cs_artifact]
    for cs_s, cs in zip(cs_signal, coeffs):
        order = np.argsort(np.abs(cs))
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

        vals_ref = np.abs(cs_s[cs_s.size // 4 : cs_s.size - cs_s.size // 4])
        ref_order = np.argsort(vals_ref)
        ref_sp = np.linspace(0, len(inv_order), len(ref_order))
        vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])
        r = vals_norm / np.abs(cs).clip(1e-10)
        cs[:] *= np.minimum(1, r)

    rec = pywt.waverec(coeffs, wavelet, mode=mode)
    return rec


def _ht(x, t):
    y = x.copy()
    y[np.abs(x) > t] = 0
    return y


def _st(x, t):
    return x.clip(-t, t)


def remove_artifact_wt(
    reference,
    artifacted,
    method="ideal_soft",
    level=None,
    mode="symmetric",
    wavelet="sym3",
):
    """Traditional wavelet thresholding denoising.

    Parameters
    ----------
    reference : numpy.ndarray
        The reference signal (i.e. the signal without artifacts).
    artifacted : numpy.ndarray
        The artifacted signal.
    method : str
        The method to use: available options are `ideal_soft`, `ideal_hard`, `sure`.
    level : int
        Maximum decomposition level. If `None`, the maximum level is chosen based on the signal length.
    """
    coeffs_ref = pywt.wavedec(reference, wavelet, level=level, mode=mode)
    coeffs_art = pywt.wavedec(artifacted, wavelet, level=level, mode=mode)

    coeffs = [c.copy() for c in coeffs_art]

    if method in ["ideal_hard", "ideal_soft"]:
        thresh_fn = _ht if method.endswith("_hard") else _st
        for cs_ref, cs in zip(coeffs_ref[1:], coeffs[1:]):
            ts = np.linspace(0, np.abs(cs).max(), 1_000)
            R = [((cs_ref - thresh_fn(cs, t)) ** 2).mean() for t in ts]
            th = ts[np.argmin(R)]
            cs[:] = thresh_fn(cs, th)
    elif method in ["universal_hard", "universal_soft"]:
        thresh_fn = _ht if method.endswith("_hard") else _st
        for cs_ref, cs in zip(coeffs_ref[1:], coeffs[1:]):
            th = cs_ref.std() * np.sqrt(2 * np.log(artifacted.size))
            cs[:] = thresh_fn(cs, th)
    elif method == "sure":
        for cs_ref, cs in zip(coeffs_ref[1:], coeffs[1:]):
            # Estimate the standard deviation (on the reference signal)
            σ = cs_ref.std()

            # SURE minimization
            ts = np.linspace(0, np.sqrt(2 * np.log(len(cs))), 1_000)
            th = σ * ts[np.argmin([sure(t, cs / σ) for t in ts])]

            # Soft thresholding
            cs[:] = cs.clip(-th, th)
    else:
        raise ValueError(f"Unknown method `{method}`")

    rec = pywt.waverec(coeffs, wavelet, mode=mode)

    return rec


def sure(t, x):
    """SURE of soft-thresholding operator (unit variance).

    Parameters
    ----------
    t : float
        Threshold value.
    x : numpy.ndarray
        Data vector.
    """
    return len(x) - 2 * (np.abs(x) < t).sum() + (np.minimum(np.abs(x), t) ** 2).sum()


# %%

sel_level = 4

rec_wt_soft1 = remove_artifact_wt(s1, x1, "ideal_soft", num_levels)
rec_wt_hard1 = remove_artifact_wt(s1, x1, "ideal_hard", num_levels)
rec_wt_sure1 = remove_artifact_wt(s1, x1, "sure", num_levels)
rec_wqn1 = remove_artifact(s1, x1, num_levels)

rec_wt_soft2 = remove_artifact_wt(s2, x2, "ideal_soft", num_levels)
rec_wt_hard2 = remove_artifact_wt(s2, x2, "ideal_hard", num_levels)
rec_wt_sure2 = remove_artifact_wt(s2, x2, "sure", num_levels)
rec_wqn2 = remove_artifact(s2, x2, num_levels)

cs_ref1 = pywt.wavedec(s1, _wavelet, level=num_levels, mode=_mode)
cs_hard1 = pywt.wavedec(
    rec_wt_hard1,
    _wavelet,
    level=num_levels,
    mode=_mode,
)
cs_soft1 = pywt.wavedec(
    rec_wt_soft1,
    _wavelet,
    level=num_levels,
    mode=_mode,
)
cs_wqn1 = pywt.wavedec(rec_wqn1, _wavelet, level=num_levels, mode=_mode)

cs_ref2 = pywt.wavedec(s2, _wavelet, level=num_levels, mode=_mode)
cs_hard2 = pywt.wavedec(
    rec_wt_hard2,
    _wavelet,
    level=num_levels,
    mode=_mode,
)
cs_soft2 = pywt.wavedec(
    rec_wt_soft2,
    _wavelet,
    level=num_levels,
    mode=_mode,
)
cs_wqn2 = pywt.wavedec(rec_wqn2, _wavelet, level=num_levels, mode=_mode)

_range = (
    -3.0 * np.max([cs_soft1[sel_level], cs_soft2[sel_level]]),
    3.0 * np.max([cs_soft1[sel_level], cs_soft2[sel_level]]),
)

print("HARD\t", ((rec_wt_hard1 - s1) ** 2).mean())
print("SOFT\t", ((rec_wt_soft1 - s1) ** 2).mean())
print("WQN\t", ((rec_wqn1 - s1) ** 2).mean())

# %%

FIG_WIDTH = 2 * 7.16

num_bins = 60

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, 2 * 0.3 * FIG_WIDTH),
    ncols=2,
    nrows=2,
)

spacing = 1.4


ax[0, 0].plot(t, s1)
# ax[0, 0].plot(t, -spacing + a)
ax[0, 0].plot(t, -1 * spacing + x1, c=COLOR_ORIG)

ax[0, 0].plot(t, -2 * spacing + rec_wt_hard1, c=COLOR_WT_HARD)
ax[0, 0].plot(t, -3 * spacing + rec_wt_soft1, c=COLOR_WT_SOFT)
ax[0, 0].plot(t, -4 * spacing + rec_wqn1, c=COLOR_WQN)

ax[0, 0].set_yticks(-np.arange(5) * spacing)
ax[0, 0].set_yticklabels(["Test signal", "Artifacted", "HT", "ST", "WQN"])

ax[0, 1].hist(
    cs_ref1[sel_level],
    bins=num_bins,
    range=_range,
    density=True,
    fc="black",
    label="Test signal",
)

ax[0, 1].hist(
    cs_hard1[sel_level],
    bins=num_bins,
    range=_range,
    density=True,
    fc=COLOR_WT_HARD,
    alpha=0.75,
    label="Hard thresholding (HT)",
)
ax[0, 1].hist(
    cs_soft1[sel_level],
    bins=num_bins,
    range=_range,
    density=True,
    alpha=0.75,
    fc=COLOR_WT_SOFT,
    label="Soft thresholding (ST)",
)
ax[0, 1].hist(
    cs_wqn1[sel_level],
    bins=num_bins,
    range=_range,
    density=True,
    fc=COLOR_WQN,
    alpha=0.75,
    label="WQN",
)
ax[0, 1].legend()

# SECOND FIGURE


ax[1, 0].plot(t, s2)
# ax[1, 0].plot(t, -spacing + a)
ax[1, 0].plot(t, -1 * spacing + x2, c=COLOR_ORIG)

ax[1, 0].plot(t, -2 * spacing + rec_wt_hard2, c=COLOR_WT_HARD)
ax[1, 0].plot(t, -3 * spacing + rec_wt_soft2, c=COLOR_WT_SOFT)
ax[1, 0].plot(t, -4 * spacing + rec_wqn2, c=COLOR_WQN)

ax[1, 0].set_yticks(-np.arange(5) * spacing)
ax[1, 0].set_yticklabels(["Test signal", "Artifacted", "HT", "ST", "WQN"])

ax[1, 1].hist(
    cs_ref2[sel_level],
    bins=num_bins,
    range=_range,
    density=True,
    fc="black",
    label="Test signal",
)

ax[1, 1].hist(
    cs_hard2[sel_level],
    bins=num_bins,
    range=_range,
    density=True,
    fc=COLOR_WT_HARD,
    alpha=0.75,
    label="Hard thresholding (HT)",
)
ax[1, 1].hist(
    cs_soft2[sel_level],
    bins=num_bins,
    range=_range,
    density=True,
    alpha=0.75,
    fc=COLOR_WT_SOFT,
    label="Soft thresholding (ST)",
)
ax[1, 1].hist(
    cs_wqn2[sel_level],
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

    rec_wt_soft1 = remove_artifact_wt(s1, x1, "ideal_soft", num_levels)
    rec_wt_hard1 = remove_artifact_wt(s1, x1, "ideal_hard", num_levels)
    rec_wt_uni_soft1 = remove_artifact_wt(s1, x1, "universal_soft", num_levels)
    rec_wt_uni_hard1 = remove_artifact_wt(s1, x1, "universal_hard", num_levels)
    # rec_wt_sure1 = remove_artifact_wt(s1, x1, "sure", num_levels)
    rec_wqn1 = remove_artifact(s1, x1, num_levels)

    rec_wt_soft2 = remove_artifact_wt(s2, x2, "ideal_soft", num_levels)
    rec_wt_hard2 = remove_artifact_wt(s2, x2, "ideal_hard", num_levels)
    rec_wt_uni_soft2 = remove_artifact_wt(s2, x2, "universal_soft", num_levels)
    rec_wt_uni_hard2 = remove_artifact_wt(s2, x2, "universal_hard", num_levels)
    # rec_wt_sure2 = remove_artifact_wt(s2, x2, "sure", num_levels)
    rec_wqn2 = remove_artifact(s2, x2, num_levels)

    mse1.append(
        {
            "wt_soft": _mse(rec_wt_soft1, s1),
            "wt_hard": _mse(rec_wt_hard1, s1),
            "wt_uni_soft": _mse(rec_wt_uni_soft1, s1),
            "wt_uni_hard": _mse(rec_wt_uni_hard1, s1),
            # "wt_sure": _mse(rec_wt_sure1, s1),
            "wqn": _mse(rec_wqn1, s1),
        }
    )

    mse2.append(
        {
            "wt_soft": _mse(rec_wt_soft2, s2),
            "wt_hard": _mse(rec_wt_hard2, s2),
            "wt_uni_soft": _mse(rec_wt_uni_soft2, s2),
            "wt_uni_hard": _mse(rec_wt_uni_hard2, s2),
            # "wt_sure": _mse(rec_wt_sure2, s2),
            "wqn": _mse(rec_wqn2, s2),
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

        rec_wt_soft = remove_artifact_wt(
            s1, x1, "ideal_soft", level=None, mode="periodization", wavelet="sym3"
        )
        rec_wt_hard = remove_artifact_wt(
            s1, x1, "ideal_hard", level=None, mode="periodization", wavelet="sym3"
        )
        rec_wt_uni_soft = remove_artifact_wt(
            s1, x1, "universal_soft", level=None, mode="periodization", wavelet="sym3"
        )
        rec_wt_uni_hard = remove_artifact_wt(
            s1, x1, "universal_hard", level=None, mode="periodization", wavelet="sym3"
        )
        rec_wqn = remove_artifact(s1, x1, None, mode="periodization", wavelet="sym3")

        vmse.append(
            {
                "wt_soft": _mse(rec_wt_soft, s1),
                "wt_hard": _mse(rec_wt_hard, s1),
                "wt_uni_soft": _mse(rec_wt_uni_soft, s1),
                "wt_uni_hard": _mse(rec_wt_uni_hard, s1),
                # "wt_sure": _mse(rec_wt_sure1, s1),
                "wqn": _mse(rec_wqn, s1),
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

rec_wt_soft = remove_artifact_wt(s1, x1, "ideal_soft", None, mode="periodization")
rec_wqn = remove_artifact(s1, x1, None, mode="periodization")

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

# plt.plot(rec_wt_soft)
plt.plot(rec_wqn)
plt.plot(remove_artifact_wt(s1, x1, "ideal_soft", 12, mode="periodization"))
plt.plot(s1)


# %% Variances

s1 = FBM(n=10_000, hurst=hurst, length=1).fbm()[: len(t)]
s1 /= s1.std()

a1 = ss.square((100 * t + np.random.randint(0, 99)) / np.pi)
a1 /= a1.std()

x1 = s1 + a1

rec_wqn = remove_artifact(s1, x1, mode="periodization", wavelet="sym3")
rec_wt_isoft = remove_artifact_wt(s1, x1, mode="periodization", wavelet="sym3")

print("WQN", _mse(rec_wqn, s1))
print("WT", _mse(rec_wt_isoft, s1))

# %%

s_coeffs = pywt.wavedec(s1, "sym3", mode="periodization")

wqn_coeffs = pywt.wavedec(rec_wqn, "sym3", mode="periodization")
wt_coeffs = pywt.wavedec(rec_wt_isoft, "sym3", mode="periodization")

plt.plot([cs.std() for cs in s_coeffs], marker="o")
plt.plot([cs.std() for cs in wqn_coeffs], marker="o")
plt.plot([cs.std() for cs in wt_coeffs], marker="o")
plt.yscale("log")

# %%

plt.plot(s1)
plt.plot(rec_wqn - 2)
plt.plot(rec_wt_isoft - 4)

# %%
