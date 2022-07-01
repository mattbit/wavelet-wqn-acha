# %%

import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

plt.style.use("hpub")
plt.rcParams["figure.figsize"] = (12, 6)

# %%

data1 = np.hstack(
    [
        np.load("/data/tmp/anastasia/190822_0_60s_trace.npy"),
        np.load("/data/tmp/anastasia/190822_60_120s_trace.npy"),
        np.load("/data/tmp/anastasia/190822_120_180s_trace.npy"),
        np.load("/data/tmp/anastasia/190822_180_240s_trace.npy"),
    ]
)

t1, x1 = data1

data2 = np.load("/data/tmp/anastasia/191203_WT_P50_LTP_slice2_poststim_0007.npy")

t2, x2 = data2

# %%

freq1 = 1 / np.diff(t1).mean()

fs1, P1 = ss.welch(x1, fs=freq1, nperseg=int(freq1 * 5))

i_max = np.argmax(fs1 > 300)
plt.plot(fs1[:i_max], P1[:i_max], label="Signal 1")

freq2 = 1 / np.diff(t2).mean()

fs2, P2 = ss.welch(x2, fs=freq2, nperseg=int(freq2 * 5))

plt.plot(fs2[:i_max], P2[:i_max], label="Signal 2")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")

plt.yscale("log")
plt.legend()
# plt.xlim(0, 200)

# %%

import pywt

W, wf = pywt.cwt(x1[:100_000], scales=np.linspace(50, 500, 200), wavelet="cmor1-2", sampling_period=1/freq1)

plt.pcolormesh(t1[:W.shape[1]], wf, np.log(np.abs(W)), vmin=0)


# %%

sos = ss.butter(2, (40, 60), fs=freq1, btype="bandpass", output="sos")
sos2 = ss.butter(2, (90, 110), fs=freq1, btype="bandpass", output="sos")
plt.plot(t1, x1 - x1.mean() - 20)
plt.plot(t1, ss.sosfiltfilt(sos, x1))
plt.plot(t1, ss.sosfiltfilt(sos2, x1))

plt.xlim(10, 10.5)
plt.ylim(-10, 20)


# %%

b, a = ss.iirnotch(51, 30, fs=freq1)
x1filt = ss.filtfilt(b, a, x1)
b, a = ss.iirnotch(100, 30, fs=freq1)
x1filt = ss.filtfilt(b, a, x1filt)

plt.plot(t1, x1)
# plt.plot(t1, x1filt)
# plt.plot(t1, np.cos(100 * t1 * (2 * np.pi)) - 30)
# plt.plot(t1, np.cos(50 * t1 * (2 * np.pi)) - 35)
plt.xlim(11, 11.5)
plt.ylim(-50, -10)

# %%

fs1, P1 = ss.welch(x1, fs=freq1, nperseg=int(freq1 * 5))

i_max = np.argmax(fs1 > 120)
plt.plot(fs1[:i_max], P1[:i_max], label="Filtered")

fs1, P1 = ss.welch(x1[:10_000 * 10], fs=freq1, nperseg=int(freq1 * 5))
plt.plot(fs1[:np.argmax(fs1 > 120)], P1[:np.argmax(fs1 > 120)], label="Partial 1")


fs1, P1 = ss.welch(x1[100_000:150_000], fs=freq1, nperseg=int(freq1 * 5))
plt.plot(fs1[:np.argmax(fs1 > 120)], P1[:np.argmax(fs1 > 120)], label="Partial 2")


plt.legend()
plt.yscale("log")

# %%
