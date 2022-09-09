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

t = np.linspace(0, 4 * 2 * np.pi, 2**14)

s = np.sin(t)
a = np.random.normal(size=s.size)

# %%

level = 10

amplitudes = np.arange(1, 50.1, 0.1)
xs = np.array([s + np.random.normal(size=s.size, scale=a) for a in amplitudes])
recs = np.array([remove_artifact(s, x, level) for x in xs])

# %%

mask = np.isclose(amplitudes, 1, atol=1e-6) | np.isclose(amplitudes, 2, atol=1e-6) | np.isclose(amplitudes, 5, atol=1e-6) | np.isclose(amplitudes, 10, atol=1e-6)
assert mask.sum() == 4

fig, axs = plt.subplots(nrows=4)
fig2, axs2 = plt.subplots(nrows=4)

for n, (x, rec, amp) in enumerate(zip(xs[mask], recs[mask], amplitudes[mask])):
    if n == 0:
        axs[n].plot(t, x, c="r", label="Artifacted signal")
        axs[n].legend()

    axs[n].plot(t, s, c="k", ls="--", label="Original")
    axs[n].plot(t, rec, c="b", label="WQN reconstruction")
    axs[n].set_title(f"$\sigma = {amp:.0f}$")
    
    if n == 1:
        axs[n].legend()

    axs2[n].plot(t, s, c="k", ls="--")
    axs2[n].plot(t, x, c="b")

fig.tight_layout()
fig.savefig("comparison_white_noise.svg")

# %%

fig, ax = plt.subplots()
ax.plot(amplitudes, ((recs - s)**2).mean(axis=-1), label="WQN corrected")

ax.plot(amplitudes, ((xs - s)**2).mean(axis=-1), c="r", label="Artifacted")

ax.set_ylabel("Mean Squared Error")
ax.set_xlabel("Noise amplitude (σ)")
ax.legend()
# plt.plot(amplitudes, amplitudes**2, c="g", ls="--", label="Square")

ax.set_yscale("log")

fig.savefig("mse_dependency_on_artifact_amplitude.svg")

# %%
