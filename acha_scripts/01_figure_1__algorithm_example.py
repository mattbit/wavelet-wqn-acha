import pywt
import numpy as np

import matplotlib.pyplot as plt

plt.style.use("resources/figstyle.mplstyle")

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

FIG_WIDTH = 2 * 7.16

# Load the data (DenoiseNet dataset)
eeg = np.load("resources/data/eeg-denoise-net/EEG_all_epochs.npy")
eog = np.load("resources/data/eeg-denoise-net/EOG_all_epochs.npy")
emg = np.load("resources/data/eeg-denoise-net/EMG_all_epochs.npy")


# Prepare the signal by artificially adding an ocular artifact
ref_slice = slice(0, 256)
art_slice = slice(256, 512)

signal = eeg[467]
signal_artifacted = signal + 2 * np.pad(eog[260][:256], (256, 0))
ts = np.arange(signal.size) / 256


# WQN method
_wrecargs = dict(wavelet="sym2", mode="symmetric")
_wdecargs = dict(**_wrecargs, level=4)

dwt_cs = pywt.wavedec(signal_artifacted, **_wdecargs)
cs_ref = [cs[: cs.size // 2] for cs in dwt_cs]
cs_art = [cs[cs.size // 2 :] for cs in dwt_cs]
cs_true = [cs[cs.size // 2 :] for cs in pywt.wavedec(signal, **_wdecargs)]

coeffs = [c.copy() for c in cs_art]
for cs_s, cs in zip(cs_ref, coeffs):
    order = np.argsort(np.abs(cs))
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))

    vals_ref = np.abs(cs_s)
    ref_order = np.argsort(vals_ref)
    ref_sp = np.linspace(0, len(inv_order) - 1, len(ref_order))
    vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

    r = vals_norm / np.abs(cs)
    cs[:] *= np.minimum(1, r)

cs_rec = coeffs.copy()
cs_rec_full = [np.concatenate([cs1, cs2]) for cs1, cs2 in zip(cs_ref, cs_rec)]

signal_rec = pywt.waverec(cs_rec_full, **_wrecargs)


# Prepare the figure
color_ref = COLORS[0]
color_art = COLORS[1]
color_wqn = COLORS[3]

fig = plt.figure(figsize=(FIG_WIDTH, 0.5 * FIG_WIDTH), constrained_layout=True)
gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.15, 1])

figA = fig.add_subfigure(
    gs[0, :],
)

axA = figA.subplots()
axA.plot(ts[ref_slice], signal[ref_slice], label="Original signal", c=color_ref)
axA.plot(ts[art_slice], signal[art_slice], c=color_ref, ls="--")

axA.plot(ts[255:], signal_artifacted[255:], label="Artifacted signal", c=color_art)
axA.plot(ts[255:], signal_rec[255:], label="WQN reconstruction", c=color_wqn)
axA.yaxis.set_visible(False)
axA.legend(loc="lower right")
axA.set_title("A. WQN-transport correction of EOG artifact", loc="left")

axA.annotate(
    "Reference interval",
    xy=[0.5, -1000],
    ha="center",
    xytext=[0.5, -1500],
    fontsize=14,
    arrowprops=dict(arrowstyle="-[, widthB=15, lengthB=0.5", lw=1),
)

axA.set_xlim(-0.05, 2.04)
axA.set_xticks([0, 0.5, 1, 1.5, 2])
axA.set_xticklabels(["0", "0.5 s", "1 s", "1.5 s", "2 s"])

# Coefficients
figB1 = fig.add_subfigure(gs[1, :])
axB1a, axB1b = figB1.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.25]))

figB2 = fig.add_subfigure(gs[2, :])
axB2a, axB2b = figB2.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.25]))

figB1.suptitle(
    "B. Transport of wavelet coefficients",
    fontsize=16,
    fontweight="semibold",
    x=0,
    ha="left",
)
figB1.supylabel("Level $c_5$", fontsize=15, fontweight="semibold", va="center", y=0.38)
figB2.supylabel("Level $c_4$", fontsize=15, fontweight="semibold", va="center", y=0.55)

cs_ref_abs = [np.abs(c) for c in cs_ref]
cs_art_abs = [np.abs(c) for c in cs_art]
cs_rec_abs = [np.abs(c) for c in cs_rec]

p = np.linspace(0, 1, cs_ref_abs[0].size)

axB1a.set_title("Cumulative density functions (CDFs)", loc="center", fontsize=15)
axB1b.set_title("Wavelet coefficients", loc="center", fontsize=15)

axB1a.plot(sorted(cs_ref_abs[0]), p, c=color_ref)
axB1a.plot(sorted(cs_art_abs[0]), p, c=color_art)
axB1a.plot(sorted(cs_rec_abs[0]), p, c=color_wqn)
axB1a.set_yticks([0, 1])
axB1a.set_ylabel("Probability", labelpad=-10)
axB1a.xaxis.set_visible(False)

axB2a.plot(sorted(cs_ref_abs[1]), p, c=color_ref)
axB2a.plot(sorted(cs_art_abs[1]), p, c=color_art)
axB2a.plot(sorted(cs_rec_abs[1]), p, c=color_wqn)
axB2a.set_yticks([0, 1])
axB2a.set_ylabel("Probability", labelpad=-10)
axB2a.set_xlabel("Amplitude")
axB2a.set_xlim(0, 5000)
axB2a.set_xticks([])

axB1b.yaxis.set_visible(False)
axB2b.yaxis.set_visible(False)

axB1b.plot(cs_true[0], c=color_ref, ls="--")
axB1b.plot(cs_art[0], c=color_art)
axB1b.plot(cs_rec[0], c=color_wqn)
axB1b.xaxis.set_visible(False)
axB1b.set_ylim(-7000, 7000)

axB2b.plot(cs_true[2], c=color_ref, ls="--")
axB2b.plot(cs_art[2], c=color_art)
axB2b.plot(cs_rec[2], c=color_wqn)
axB2b.set_ylim(-1500, 1500)
axB2b.set_xticks([0, cs_ref[2].size // 2, cs_ref[2].size])
axB2b.set_xticklabels(["1 s", "1.5 s", "2 s"])
axB2b.set_xlim(-1.5, 34.5)

cdf_ref = sorted(cs_ref_abs[0])
cdf_art = sorted(cs_art_abs[0])

axB1a.annotate(
    None,
    xy=(cdf_ref[10] + 72, p[10]),
    xytext=(cdf_art[10] - 72, p[10]),
    ha="center",
    va="center",
    arrowprops=dict(width=2, headlength=6, headwidth=8, color=color_wqn),
)
axB1a.scatter(
    [cdf_ref[10], cdf_art[10]], [p[10], p[10]], c=[color_wqn, color_art], s=20
)
axB1a.text(
    (cdf_ref[10] + cdf_art[10]) / 2,
    p[10] + 0.05,
    "$T_5$",
    va="bottom",
    c=color_wqn,
    fontsize=15,
)

for n, (c_art, c_rec) in enumerate(zip(cs_art[0], cs_rec[0])):
    if abs(c_art - c_rec) > 2000:
        axB1b.annotate(
            "",
            xy=(n, c_rec + np.sign(c_rec) * 600),
            xytext=(n, c_art - np.sign(c_art) * 600),
            arrowprops=dict(width=1, headlength=6, headwidth=6, color=color_wqn),
        )

fig.savefig("./output/acha_fig1_algorithm_example.pdf")
print("Figure saved to `./output/acha_fig1_algorithm_example.pdf`")
