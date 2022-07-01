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

# %%

import h5py

f = h5py.File("data/01_physiobank.h5", "r")

eeg = f["eeg_1"]["eeg_reference"][0, :] * 1e3

eog = np.load("data/eeg-denoise-net/EOG_all_epochs.npy")

np.random.seed(3883)
eog_ = eog.reshape(-1)

hurst = 0.5
N_samples = 10_000

t = np.arange(N_samples) / 256
s = eeg[:len(t)]
# s = FBM(n=N_samples, hurst=hurst, length=1).fbm()[: len(t)]
a = eog_[:N_samples]
a *= 120 / a.max()
# a = ss.square(100 * t / np.pi)

x = s + a

plt.plot(t, x)
plt.plot(t, s)
plt.xlim(0, 20)
plt.ylim(-100, 100)


# %%

_wavelet = "db3"
_mode = "symmetric"


def remove_artifact(reference, artifact, level=None):
    cs_signal = pywt.wavedec(reference, _wavelet, level=level, mode=_mode)
    cs_artifact = pywt.wavedec(artifact, _wavelet, level=level, mode=_mode)

    coeffs = [c.copy() for c in cs_artifact]
    for cs_s, cs in zip(cs_signal, coeffs):
        order = np.argsort(np.abs(cs))
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

        vals_ref = np.abs(cs_s[cs_s.size // 4 : cs_s.size - cs_s.size // 4])
        ref_order = np.argsort(vals_ref)
        ref_sp = np.linspace(0, len(inv_order), len(ref_order))
        vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

        r = vals_norm / np.abs(cs)
        cs[:] *= np.minimum(1, r)
        # cs[:] *= r

    rec = pywt.waverec(coeffs, _wavelet, mode=_mode)
    return rec


def remove_artifact_wt(reference, artifact, threshold="hard", level=None):
    coeffs_ref = pywt.wavedec(reference, _wavelet, level=level, mode=_mode)
    coeffs_art = pywt.wavedec(artifact, _wavelet, level=level, mode=_mode)

    coeffs = [c.copy() for c in coeffs_art]
    for cs_ref, cs in zip(coeffs_ref, coeffs):
        th = np.abs(cs_ref[cs_ref.size // 4 : cs_ref.size - cs_ref.size // 4]).max()
        if threshold == "hard":
            cs[np.abs(cs) > th] = 0
        elif threshold == "soft":
            cs[:] = cs.clip(-th, th)

    rec = pywt.waverec(coeffs, _wavelet, mode=_mode)
    return rec


# %%

num_levels = 9

fig, ax = plt.subplots()
ax.plot(t, x, label="Artifacted signal", c="k")
# ax.plot(t, s, label="Original signal", c="k", ls="--")
ax.plot(
    t,
    remove_artifact_wt(s, x, "hard", num_levels),
    c="r",
    label="Hard-thresholding",
    lw=1.5,
)
ax.plot(
    t,
    remove_artifact_wt(s, x, "soft", num_levels),
    c="orange",
    label="Soft-thresholding",
    lw=1.5,
)
ax.plot(t, remove_artifact(s, x, num_levels), c="navy", label="WQN", lw=1.5)
# ax.legend()
# fig.savefig("../figures/theory/example_cusp_3x.svg")
# fig.savefig("../figures/theory/example_cusp_3x.eps")

# plt.plot(t, remove_artifact_ht(x, 16, 4), c="r")
# plt.xlim(0.8, 1.2)

# %%

cs_ref = pywt.wavedec(s, _wavelet, level=num_levels, mode=_mode)
cs_hard = pywt.wavedec(
    remove_artifact_wt(s, x, "hard", num_levels), _wavelet, level=num_levels, mode=_mode
)
cs_soft = pywt.wavedec(
    remove_artifact_wt(s, x, "soft", num_levels), _wavelet, level=num_levels, mode=_mode
)
cs_wqn = pywt.wavedec(
    remove_artifact(s, x, num_levels), _wavelet, level=num_levels, mode=_mode
)


_range = (-1.1 * cs_soft[5].max(), 1.1 * cs_soft[5].max())
plt.hist(cs_ref[5], bins=40, range=_range, density=True, label="Reference")
plt.hist(cs_wqn[5], bins=40, range=_range, density=True, alpha=0.75, label="WQN")
plt.hist(
    cs_hard[5],
    bins=40,
    range=_range,
    density=True,
    alpha=0.75,
    label="Hard thresholding",
)
plt.hist(
    cs_soft[5],
    bins=40,
    range=_range,
    density=True,
    alpha=0.75,
    label="Soft thresholding",
)
plt.legend()


# %%

FIG_WIDTH = 2 * 7.16

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, 0.3 * FIG_WIDTH),
    ncols=2,
)


spacing = 100

ax[0].plot(t, s)
# ax[0].plot(t, -spacing + a)
ax[0].plot(t, -1 * spacing + x, c=COLORS[6])

ax[0].plot(t, -2 * spacing + remove_artifact_wt(s, x, "hard", num_levels), c=COLORS[7])
ax[0].plot(t, -3 * spacing + remove_artifact_wt(s, x, "soft", num_levels), c=COLORS[3])
ax[0].plot(t, -4 * spacing + remove_artifact(s, x, num_levels), c=COLORS[5])

ax[0].set_yticks(-np.arange(5) * spacing)
ax[0].set_yticklabels(
    ["Reference", "Artifacted", "HT", "ST", "WQN"]
)

_range = (-1.1 * cs_soft[5].max(), 1.1 * cs_soft[5].max())
ax[1].hist(
    cs_ref[5],
    bins=40,
    range=_range,
    density=True,
    fc="black",
    label="Reference",
)

ax[1].hist(
    cs_hard[5],
    bins=40,
    range=_range,
    density=True,
    fc=COLORS[7],
    alpha=0.75,
    label="Hard thresholding (HT)",
)
ax[1].hist(
    cs_soft[5],
    bins=40,
    range=_range,
    density=True,
    alpha=0.75,
    fc=COLORS[3],
    label="Soft thresholding (ST)",
)
ax[1].hist(
    cs_wqn[5],
    bins=40,
    range=_range,
    density=True,
    fc=COLORS[5],
    alpha=0.75,
    label="WQN",
)
ax[1].legend()

ax[0].set_title("A. Time-domain signals", loc="left")
ax[1].set_title("B. Distribution of wavelet coefficients", loc="left")

ax[0].set_xlabel("Time")
ax[1].yaxis.set_visible(False)
ax[1].set_xlabel("Coefficient value")
ax[0].set_xlim(0, 30)
fig.tight_layout()


fig.savefig(f"/home/matteo/Thesis/Resources/wqn_comparison.svg")

# %%


fig, ax = plt.subplots(figsize=(0.5 * FIG_WIDTH, 0.5 * 0.3 * FIG_WIDTH))
ax.plot(t, remove_artifact_wt(s, x, "soft", num_levels), c=COLORS[3])
ax.yaxis.set_visible(False)
ax.set_xlabel("Time")
ax.set_ylim(-100, 100)
fig.savefig(f"/home/matteo/Thesis/Resources/wqn_comparison_wt_signal.svg")

fig, ax = plt.subplots(figsize=(0.5 * FIG_WIDTH, 0.5 * 0.3 * FIG_WIDTH))
ax.hist(
    cs_soft[5],
    bins=40,
    range=_range,
    density=True,
    alpha=0.75,
    fc=COLORS[3],
)
ax.yaxis.set_visible(False)
ax.set_xlabel("Coefficient value")
fig.savefig(f"/home/matteo/Thesis/Resources/wqn_comparison_wt_hist.svg")

# %%

fig, ax = plt.subplots(figsize=(0.5 * FIG_WIDTH, 0.5 * 0.3 * FIG_WIDTH))
ax.plot(t, s, c=COLORS[0])
ax.yaxis.set_visible(False)
ax.set_xlabel("Time")
ax.set_ylim(-100, 100)
fig.savefig(f"/home/matteo/Thesis/Resources/wqn_comparison_ref_signal.svg")

fig, ax = plt.subplots(figsize=(0.5 * FIG_WIDTH, 0.5 * 0.3 * FIG_WIDTH))
ax.hist(
    cs_ref[5],
    bins=40,
    range=_range,
    density=True,
    alpha=0.75,
    fc=COLORS[0],
)
ax.yaxis.set_visible(False)
ax.set_xlabel("Coefficient value")
fig.savefig(f"/home/matteo/Thesis/Resources/wqn_comparison_ref_hist.svg")

# %%

fig, ax = plt.subplots(figsize=(0.5 * FIG_WIDTH, 0.5 * 0.3 * FIG_WIDTH))
ax.plot(t, remove_artifact(s, x, num_levels), c=COLORS[5])
ax.yaxis.set_visible(False)
ax.set_xlabel("Time")
ax.set_ylim(-100, 100)
fig.savefig(f"/home/matteo/Thesis/Resources/wqn_comparison_wqn_signal.svg")


fig, ax = plt.subplots(figsize=(0.5 * FIG_WIDTH, 0.5 * 0.3 * FIG_WIDTH))
ax.hist(
    cs_wqn[5],
    bins=40,
    range=_range,
    density=True,
    alpha=0.75,
    fc=COLORS[5],
)
ax.yaxis.set_visible(False)
ax.set_xlabel("Coefficient value")
fig.savefig(f"/home/matteo/Thesis/Resources/wqn_comparison_wqn_hist.svg")

# %%
