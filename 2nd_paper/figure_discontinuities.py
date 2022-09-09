# %%

import pywt
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss

from methods import WQNDenoiser, WTDenoiser

plt.style.use("hpub")


_wavelet = "db5"
_mode = "periodic"


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

plt.style.use("hpub")

from scipy.signal import sawtooth

t = np.linspace(0, 2 * 2 * np.pi, 2**14)

s = np.sin(t)
a = -sawtooth(2 * (t + np.pi), 0.5)

x = s + a

plt.plot(t, s)
plt.plot(t, x, c="r")

# %%
