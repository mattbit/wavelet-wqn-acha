# %%

import pywt
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

from methods import WQNDenoiser, WTDenoiser

plt.style.use("hpub")


_wavelet = "db2"
_mode = "smooth"


def remove_artifact(reference, artifact, level=None, with_coeffs=False):
    cs_signal = pywt.wavedec(reference, _wavelet, level=level, mode=_mode)
    cs_artifact = pywt.wavedec(artifact, _wavelet, level=level, mode=_mode)

    coeffs = [c.copy() for c in cs_artifact]
    for cs_s, cs in zip(cs_signal, coeffs):
        order = np.argsort(np.abs(cs))
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

        vals_ref = np.abs(cs_s)
        ref_order = np.argsort(vals_ref)
        ref_sp = np.linspace(0, len(inv_order), len(ref_order))
        vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

        r = vals_norm / np.abs(cs).clip(1e-30)
        cs[:] *= np.minimum(1, r)
        # cs[:] *= r

    rec = pywt.waverec(coeffs, _wavelet, mode=_mode)

    if with_coeffs:
        return rec, coeffs

    return rec


def remove_artifact_wt(reference, artifact, threshold="hard", level=None):
    coeffs_ref = pywt.wavedec(reference, _wavelet, level=level, mode=_mode)
    coeffs_art = pywt.wavedec(artifact, _wavelet, level=level, mode=_mode)

    coeffs = [c.copy() for c in coeffs_art]
    for cs_ref, cs in zip(coeffs_ref, coeffs):
        th = np.abs(cs_ref).max()
        if threshold == "hard":
            cs[np.abs(cs) > th] = 0
        elif threshold == "soft":
            cs[:] = cs.clip(-th, th)

    rec = pywt.waverec(coeffs, _wavelet, mode=_mode)
    return rec


# %% Ramp + square wave

t = np.linspace(0, 3, 10000)

s = 1 * t
# a = ((t > 1) & (t < 2))
a = (
    2
    * (1 * ((t >= 1) & (t < 1.5)) - 1 * ((t >= 1.5) & (t < 2))).cumsum()
    / (t.size / 3)
)

art_amplitude = 3

x = s + art_amplitude * a

plt.plot(t, x)
plt.plot(t, s)


# %%

num_levels = 4

fig, ax = plt.subplots()
ax.plot(t, x, label="Artifacted signal", c="k")
ax.plot(t, s, label="Original signal", c="k", ls="--")
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

wqn_rec, wqn_coeffs = remove_artifact(s, x, num_levels, with_coeffs=True)
ax.plot(t, wqn_rec, c="navy", label="WQN", lw=1.5)
ax.legend()

# %%

coeffs = pywt.wavedec(x, _wavelet, level=num_levels, mode=_mode)
coeffs_ref = pywt.wavedec(s, _wavelet, level=num_levels, mode=_mode)

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)

ax0.plot(coeffs[0], c="k", label="Artifacted")
ax0.plot(coeffs_ref[0], c="k", ls="--")
ax0.plot(wqn_coeffs[0], c="navy", label="WQN")
ax0.set_ylabel(f"cA{num_levels}")

ax1.plot(coeffs[1], c="k", label="Artifacted")
ax1.plot(coeffs_ref[1], c="k", ls="--")
ax1.plot(wqn_coeffs[1], c="navy", label="WQN")
ax1.set_ylabel(f"cD{num_levels}")

ax2.plot(coeffs[2], c="k", label="Artifacted")
ax2.plot(coeffs_ref[2], c="k", ls="--")
ax2.plot(wqn_coeffs[2], c="navy", label="WQN")
ax2.set_ylabel(f"cD{num_levels -1}")

fig.suptitle(f"Cusp wavelet coefficients (wavelet={_wavelet},level={num_levels},cusp_amplitude={art_amplitude})")
fig.savefig(f"cusp_wcoeffs_wavelet-{_wavelet}__level-{num_levels}__cusp_amplitude-{art_amplitude}.svg")


# %%

ref_cdf = list(sorted(coeffs_ref[0]))
art_cdf = list(sorted(coeffs[0]))
wqn_cdf = list(sorted(wqn_coeffs[0]))

fig, ax = plt.subplots()
ax.plot(ref_cdf, np.linspace(0, 1, len(ref_cdf)), c='k', ls="--", lw=2, label="Original")
ax.plot(art_cdf, np.linspace(0, 1, len(art_cdf)), c='k', lw=2, label="Artifacted")
ax.plot(wqn_cdf, np.linspace(0, 1, len(wqn_cdf)), label="WQN", c="navy", lw=2)

ax.set_ylabel("Probability")
ax.set_xlabel("Coefficient value")
ax.legend()

fig.suptitle(f"Cusp CDF (wavelet={_wavelet},level={num_levels},cusp_amplitude={art_amplitude})")
fig.savefig(f"cusp_cdf_wavelet-{_wavelet}__level-{num_levels}__cusp_amplitude-{art_amplitude}.svg")

# %%

