import pywt
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

plt.style.use('hpub')
plt.rcParams['figure.figsize'] = (15, 6)


def remove_artifact(reference, artifact):
    cs_signal = pywt.wavedec(reference, "sym5", mode="periodization")
    cs_artifact = pywt.wavedec(artifact, "sym5", mode="periodization")

    coeffs = [c.copy() for c in cs_artifact]
    for cs_s, cs in zip(cs_signal, coeffs):
        order = np.argsort(np.abs(cs))
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

        vals_ref = np.abs(cs_s)
        ref_order = np.argsort(vals_ref)
        ref_sp = np.linspace(0, len(inv_order) - 1, len(ref_order))
        vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

        r = vals_norm / np.abs(cs)
        cs[:] *= np.minimum(1, r)

    cs_reconstruction = coeffs

    rec = pywt.waverec(coeffs, "sym5", mode="periodization")
    return rec


def remove_artifact_swt(reference, artifact):
    cs_signal = pywt.swt(reference, "sym5", norm=True, trim_approx=True)
    cs_artifact = pywt.swt(artifact, "sym5", norm=True, trim_approx=True)

    coeffs = [c.copy() for c in cs_artifact]
    for cs_s, cs in zip(cs_signal, coeffs):
        order = np.argsort(np.abs(cs))
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

        vals_ref = np.abs(cs_s)
        ref_order = np.argsort(vals_ref)
        ref_sp = np.linspace(0, len(inv_order) - 1, len(ref_order))
        vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

        r = vals_norm / np.abs(cs)
        cs[:] *= np.minimum(1, r)
        # cs[:] *= r

    cs_reconstruction = coeffs

    rec = pywt.iswt(coeffs, "sym5", norm=True)
    return rec

# %%

ts = np.linspace(0, 100 * np.pi, 10000)

wave_ref = np.cos(ts)
wave_art = wave_ref + ts * np.sin(1 /( 1e-3 * (1 + ts))) / 50

np.sin(ts) + 5 * np.random.normal(size=ts.size)
# wave_art = ss.sawtooth(ts + 0.5 * np.pi, width=0.5)
# wave_ref = ss.sawtooth(ts + 0.5 * np.pi, width=0.5)
# wave_art = 0.25 * np.sin(ts) + 0.25 * ss.sawtooth(ts + 0.5 * np.pi, width=0.5)

plt.plot(ts / (2 * np.pi), wave_ref, label="Reference", ls="--")
plt.plot(ts / (2 * np.pi), wave_art, c='r', label="Artifact")
plt.plot(ts / (2 * np.pi), remove_artifact_swt(wave_ref, wave_art), c='navy', label="Restored")
plt.legend()
# plt.xlim(0, 10)

%matplotlib qt

# plt.savefig("figures/analytical_examples/sin_square_added_swt.svg")

# %%

reference = wave_ref
artifact = wave_art
cs_signal = pywt.swt(reference, "sym5", norm=True, trim_approx=True)
cs_artifact = pywt.swt(artifact, "sym5", norm=True, trim_approx=True)

coeffs = [c.copy() for c in cs_artifact]
for cs_s, cs in zip(cs_signal, coeffs):
    order = np.argsort(np.abs(cs))
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))

    vals_ref = np.abs(cs_s)
    ref_order = np.argsort(vals_ref)
    ref_sp = np.linspace(0, len(inv_order) - 1, len(ref_order))
    vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

    r = vals_norm / np.abs(cs)
    # cs[:] *= np.minimum(1, r)
    # cs[:] *= r

    plt.plot(ts, cs_s, label="Reference", ls="--")
    plt.plot(ts, cs, label="Original")
    plt.plot(ts, cs * np.minimum(1, r), label="Corrected")
    plt.legend()
    plt.xlim(0, 4)
    plt.show()

# cs_reconstruction = coeffs
#
# rec = pywt.iswt(coeffs, "sym5", norm=True)
