import pywt
import numpy as np

from abc import abstractmethod
from tools import intervals_to_mask


class SingleChannelDenoiser:
    def run(self, signal, artifacts, fs=None, reference=None):
        norm_signal, norm_params = self.normalize(signal)

        if reference is not None:
            s_mean, s_std = norm_params
            norm_ref = (reference - s_mean) / s_std
        else:
            norm_ref = [None] * signal.shape[0]

        filtered = np.zeros_like(signal)

        for n in range(norm_signal.shape[0]):
            filtered[n] = self.run_single_channel(
                norm_signal[n], artifacts, fs, norm_ref[n]
            )

        return self.denormalize(filtered, norm_params)

    def normalize(self, signal):
        s_mean = signal.mean(axis=-1).reshape(-1, 1)
        s_std = signal.std(axis=-1).reshape(-1, 1)

        return (
            (signal - s_mean) / s_std,
            (s_mean, s_std),
        )

    def denormalize(seld, signal, params):
        s_mean, s_std = params
        return signal * s_std + s_mean

    def run_single_channel(signal, artifacts):
        raise NotImplementedError()


class SWTDenoiser(SingleChannelDenoiser):
    def __init__(self, wavelet="sym4", level=5):
        self.wavelet = wavelet
        self.level = level

    def pad(self, data):
        min_div = 2**self.level
        remainder = len(data) % min_div
        pad_len = (min_div - remainder) % min_div

        return np.pad(data, (0, pad_len))


class WaveletThresholding(SWTDenoiser):
    def __init__(self, wavelet="sym4", level=5, mode="hard"):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def run_single_channel(self, signal, artifacts, fs=None, reference=None):
        sig_ = self.pad(signal)
        coeffs = pywt.swt(sig_, self.wavelet, self.level, norm=True, trim_approx=True)
        coeffs = np.array(coeffs)

        artifact_mask = intervals_to_mask(artifacts, coeffs.shape[1])

        k = np.sqrt(2 * np.log(coeffs.shape[1]))
        thresholds = k * np.median(np.abs(coeffs), axis=1) / 0.6745

        for ws, th in zip(coeffs, thresholds):
            ws[artifact_mask] = self.threshold(ws[artifact_mask], th)

        rec = pywt.iswt(coeffs, wavelet=self.wavelet, norm=True)

        return rec[: len(signal)]

    def threshold(self, coeffs, threshold):
        if self.mode == "hard":
            return np.where(np.abs(coeffs) <= threshold, coeffs, 0.0)
        elif self.mode == "soft":
            return np.clip(coeffs, -threshold, threshold)

        raise RuntimeError(f"Invalid thresholding mode `{self.mode}`.")


class WaveletQuantileNormalization(SingleChannelDenoiser):
    def __init__(self, wavelet="sym4", mode="periodization", alpha=1, n=30):
        self.wavelet = wavelet
        self.alpha = alpha
        self.mode = mode
        self.n = n

    def run_single_channel(self, signal, artifacts, fs=None, reference=None):
        restored = signal.copy()

        for n, (i, j) in enumerate(artifacts):
            min_a = 0
            max_b = signal.size

            if n > 0:
                min_a = artifacts[n - 1][1]
            if n + 1 < len(artifacts):
                max_b = artifacts[n + 1][0]

            size = j - i

            level = int(np.log2(size / self.n))

            if level < 1:
                continue

            # level = pywt.dwt_max_level(size, self.wavelet) - 1

            ref_size = max(self.n * 2**level, size)
            a = max(min_a, i - ref_size)
            b = min(max_b, j + ref_size)

            coeffs = pywt.wavedec(
                signal[a:b], self.wavelet, mode=self.mode, level=level
            )

            for cs in coeffs:
                k = int(np.round(np.log2(b - a) - np.log2(cs.size)))
                ik, jk = np.array([i - a, j - a]) // 2**k

                refs = [cs[:ik], cs[jk:]]
                if len(refs[0]) == 0 and len(refs[1]) == 0:
                    continue

                order = np.argsort(np.abs(cs[ik:jk]))
                inv_order = np.empty_like(order)
                inv_order[order] = np.arange(len(order))

                vals_ref = np.abs(np.concatenate(refs))
                ref_order = np.argsort(vals_ref)
                ref_sp = np.linspace(0, len(inv_order), len(ref_order))
                vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

                r = vals_norm / np.abs(cs[ik:jk])

                cs[ik:jk] *= np.minimum(1, r) ** self.alpha

            rec = pywt.waverec(coeffs, self.wavelet, mode=self.mode)
            restored[i:j] = rec[i - a : j - a]

        return restored


# %%


class WaveletDenoiser:
    def __init__(self, mode="symmetric", wavelet="sym3", max_level=None):
        self.mode = mode
        self.wavelet = wavelet
        self.max_level = max_level

    def _dwt(self, x):
        return pywt.wavedec(x, self.wavelet, level=self.max_level, mode=self.mode)

    def _idwt(self, coeffs):
        return pywt.waverec(coeffs, self.wavelet, mode=self.mode)

    @abstractmethod
    def denoise(self, signal, reference):
        """Denoise the artifacted signal based on a clean reference.

        Parameters
        ----------
        signal : numpy.ndarray
            Artifacted signal.
        reference : numpy.ndarray
            Clean reference signal.

        Returns
        -------
            numpy.ndarray : The denoised signal.
        """


class WQNDenoiser(WaveletDenoiser):
    def denoise(self, signal, reference, with_coeffs=False):
        cs_reference = self._dwt(reference)
        cs_artifact = self._dwt(signal)

        coeffs = [c.copy() for c in cs_artifact]
        for cs_ref, cs in zip(cs_reference, coeffs):
            order = np.argsort(np.abs(cs))
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(len(order))

            vals_ref = np.abs(cs_ref[cs_ref.size // 4 : cs_ref.size - cs_ref.size // 4])
            ref_order = np.argsort(vals_ref)
            ref_sp = np.linspace(0, len(inv_order), len(ref_order))
            vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])
            r = vals_norm / np.abs(cs).clip(1e-10)
            cs[:] *= np.minimum(1, r)

        if with_coeffs:
            return self._idwt(coeffs), coeffs

        return self._idwt(coeffs)


class WTDenoiser(WaveletDenoiser):
    _available_methods = [
        "ideal_hard",
        "ideal_soft",
        "universal_hard",
        "universal_soft",
        "sure",
    ]

    def __init__(self, method, mode="symmetric", wavelet="sym3", max_level=None):
        super().__init__(mode, wavelet, max_level)

        if method not in self._available_methods:
            raise ValueError(f"Unknown method `{self.method}`")

        self.method = method

    def denoise(self, signal, reference, with_coeffs=False):
        """Denoise the artifacted signal based on a clean reference.

        Parameters
        ----------
        signal : numpy.ndarray
            Artifacted signal.
        reference : numpy.ndarray
            Clean reference signal.

        Returns
        -------
            numpy.ndarray : The denoised signal.
        """
        coeffs_ref = self._dwt(reference)
        coeffs_art = self._dwt(signal)

        coeffs = [c.copy() for c in coeffs_art]
        thresh_fn = _ht if self.method.endswith("_hard") else _st
        print(thresh_fn)
        if self.method in ["ideal_hard", "ideal_soft"]:
            # Ideal thresholding minimizing MSE
            for cs_ref, cs in zip(coeffs_ref[1:], coeffs[1:]):
                ts = np.linspace(0, np.abs(cs).max(), 1_000)
                R = [((cs_ref - thresh_fn(cs, t)) ** 2).mean() for t in ts]
                th = ts[np.argmin(R)]
                cs[:] = thresh_fn(cs, th)
        elif self.method in ["universal_hard", "universal_soft"]:
            # Universal thresholding σ √(2 log N), where the σ is calculated on
            # the non-artifacted signal (reference).
            for cs_ref, cs in zip(coeffs_ref[1:], coeffs[1:]):
                th = cs_ref.std() * np.sqrt(2 * np.log(len(signal)))
                cs[:] = thresh_fn(cs, th)
        elif self.method == "sure":
            for cs_ref, cs in zip(coeffs_ref[1:], coeffs[1:]):
                # Estimate the standard deviation (on the reference signal)
                σ = cs_ref.std()

                # SURE minimization
                ts = np.linspace(0, np.sqrt(2 * np.log(len(cs))), 1_000)
                th = σ * ts[np.argmin([_sure(t, cs / σ) for t in ts])]

                # Soft thresholding
                cs[:] = _st(cs, th)
        else:
            raise ValueError(f"Unknown method `{self.method}`")

        if with_coeffs:
            return self._idwt(coeffs), coeffs

        return self._idwt(coeffs)


def _sure(t, x):
    """SURE of soft-thresholding operator (unit variance).

    Parameters
    ----------
    t : float
        Threshold value.
    x : numpy.ndarray
        Data vector.
    """
    return len(x) - 2 * (np.abs(x) < t).sum() + (np.minimum(np.abs(x), t) ** 2).sum()


def _ht(x, t):
    """Hard thresholding."""
    return np.where(np.abs(x) <= t, x, 0)


def _st(x, t):
    """Soft thresholding."""
    return x.clip(-t, t)
