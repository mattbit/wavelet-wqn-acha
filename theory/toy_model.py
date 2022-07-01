# %%
import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from fbm import FBM

plt.style.use("hpub")
plt.rcParams["figure.figsize"] = (18, 9)


f = FBM(n=10_000, hurst=0.5, length=1)
realization = f.fbm()
plt.plot(realization)

# %%

t = np.linspace(0, 3, 10000)
# s = t
# s = 1 * np.random.normal(size=t.size)
s = realization[:10_000]
# a = 10 * (1*((t >= 1.4) & (t < 1.5)) - 1*((t >= 1.5) & (t < 1.6))).cumsum() / (t.size / 3)
a = ss.square(100 * t / np.pi)
# a = np.cos(11 * t * np.pi / 3)
# 3 * ((t >= 4.5) & (t < 5.5))
x = s + a

plt.plot(t, x)
plt.plot(t, s)

# %%

_wavelet = "db3"
_mode = "smooth"

def remove_artifact(reference, artifact, level=None):
    cs_signal = pywt.wavedec(reference, _wavelet, level=level, mode=_mode)
    cs_artifact = pywt.wavedec(artifact, _wavelet, level=level, mode=_mode)

    coeffs = [c.copy() for c in cs_artifact]
    for cs_s, cs in zip(cs_signal, coeffs):
        order = np.argsort(np.abs(cs))
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

        vals_ref = np.abs(cs_s[cs_s.size // 4: cs_s.size - cs_s.size // 4])
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
        th = np.abs(cs_ref[cs_ref.size // 4: cs_ref.size - cs_ref.size // 4]).max()
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
ax.plot(t, remove_artifact_wt(s, x, "hard", num_levels), c="r",  label="Hard-thresholding", lw=1.5)
ax.plot(t, remove_artifact_wt(s, x, "soft", num_levels), c="orange", label="Soft-thresholding", lw=1.5)
ax.plot(t, remove_artifact(s, x, num_levels), c="navy", label="WQN", lw=1.5)
# ax.legend()
# fig.savefig("../figures/theory/example_cusp_3x.svg")
# fig.savefig("../figures/theory/example_cusp_3x.eps")

# plt.plot(t, remove_artifact_ht(x, 16, 4), c="r")
# plt.xlim(0.8, 1.2)

# %%

cs_ref = pywt.wavedec(s, _wavelet, level=num_levels, mode=_mode)
cs_hard = pywt.wavedec(remove_artifact_wt(s, x, "hard", num_levels), _wavelet, level=num_levels, mode=_mode)
cs_soft = pywt.wavedec(remove_artifact_wt(s, x, "soft", num_levels), _wavelet, level=num_levels, mode=_mode)
cs_wqn = pywt.wavedec(remove_artifact(s, x, num_levels), _wavelet, level=num_levels, mode=_mode)

_range = (-0.3, 0.3)
plt.hist(cs_ref[5], bins=40, range=_range, density=True, label="Reference")
plt.hist(cs_wqn[5], bins=40, range=_range, density=True, alpha=0.75, label="WQN")
plt.hist(cs_hard[5], bins=40, range=_range, density=True, alpha=0.75, label="Hard thresholding")
plt.hist(cs_soft[5], bins=40, range=_range, density=True, alpha=0.75, label="Soft thresholding")
plt.legend()


# %%


signal = s
signal_artifacted = s + a

cs_signal = pywt.wavedec(signal, _wavelet, level=num_levels, mode=_mode)
cs_artifact = pywt.wavedec(signal_artifacted, _wavelet, level=num_levels, mode=_mode)
cs_reconstruction = pywt.wavedec(remove_artifact(s, x, num_levels), _wavelet, level=num_levels, mode=_mode)
# cs_reconstruction = pywt.wavedec(remove_artifact_wt(s, x, "soft", num_levels), _wavelet, level=num_levels, mode=_mode)

level_labels = [f"cA{num_levels}"] + [f"cD{n}" for n in range(num_levels, 0, -1)]

fig, axes = plt.subplots(nrows=num_levels + 1, ncols=2, figsize=(20, 18))

ts = t

for n, (cs_ref, cs_art, cs_rec, level_label) in enumerate(zip(cs_signal, cs_artifact, cs_reconstruction, level_labels)):
    cs_ref_abs = np.abs(cs_ref)
    cs_art_abs = np.abs(cs_art)
    cs_rec_abs = np.abs(cs_rec)

    p = np.linspace(0, 1, cs_ref_abs.size)

    axes[n, 0].plot(sorted(cs_ref_abs), p, label="Reference signal")
    axes[n, 0].plot(sorted(cs_art_abs), p, label="Artifacted")
    axes[n, 0].plot(sorted(cs_rec_abs), p, label="Reconstruction")
    axes[n, 0].set_ylabel(level_label)

    cs_proj_rec = [np.zeros_like(cs_) for cs_ in cs_reconstruction]
    cs_proj_art = [np.zeros_like(cs_) for cs_ in cs_reconstruction]
    cs_proj_ref = [np.zeros_like(cs_) for cs_ in cs_reconstruction]

    cs_proj_rec[n][:] = cs_rec
    cs_proj_art[n][:] = cs_art
    cs_proj_ref[n][:] = cs_ref

    proj_rec = pywt.waverec(cs_proj_rec, _wavelet)
    proj_ref = pywt.waverec(cs_proj_ref, _wavelet)
    proj_art = pywt.waverec(cs_proj_art, _wavelet)

    # axes[n, 1].plot(ts, proj_ref, label="Reference signal")
    # axes[n, 1].plot(ts, proj_art, label="Artifacted")
    # axes[n, 1].plot(ts, proj_rec, label="Reconstruction")
    axes[n, 1].plot(np.abs(cs_ref), label="Reference signal")
    axes[n, 1].plot(np.abs(cs_art), label="Artifacted")
    axes[n, 1].plot(np.abs(cs_rec), label="Reconstruction")

    axes[n, 0].sharex(axes[0, 0])
    # axes[n, 1].sharex(axes[0, 1])
    axes[n, 0].xaxis.set_visible(False)
    # axes[n, 1].xaxis.set_visible(False)


axes[0, 0].legend()
axes[-1, 0].xaxis.set_visible(True)
axes[-1, 1].xaxis.set_visible(True)
axes[-1, 0].set_xlabel("Amplitude")
# axes[-1, 1].set_xlabel("Time (s)")
# %%
