import ast
import numpy as np
import pandas as pd
from fbm import FBM
from scipy import stats
from tqdm import trange
from pathlib import Path
import scipy.signal as ss

from methods import WQNDenoiser, WTDenoiser, ZeroDenoiser, mse


# Prepare data and denoisers
a_type = "square"
hurst = 0.5

f_art = getattr(ss, a_type)

denoiser_wqn = WQNDenoiser(wavelet="sym3", mode="periodization")
denoiser_wt_isoft = WTDenoiser(
    method="ideal_soft", wavelet="sym3", mode="periodization"
)
denoiser_wt_ihard = WTDenoiser(
    method="ideal_hard", wavelet="sym3", mode="periodization"
)
denoiser_wt_uni = WTDenoiser(
    method="universal_soft", wavelet="sym3", mode="periodization"
)
denoiser_wt_sure = WTDenoiser(method="sure", wavelet="sym3", mode="periodization")
denoiser_zero = ZeroDenoiser(wavelet="sym3", mode="periodization")

denoisers = [
    ("wqn", denoiser_wqn),
    ("wt_ihard", denoiser_wt_ihard),
    ("wt_isoft", denoiser_wt_isoft),
    ("wt_uni", denoiser_wt_uni),
    ("wt_sure", denoiser_wt_sure),
    ("wt_zero", denoiser_zero),
]


# Table of results

scale = 2  # scale of the artifact
num_realizations = 1_000
t = np.linspace(0, 3, 2**12)

# We compute the MSE just in the central part of the signal to avoid
# considering border effects.
mse_slice = slice(t.size // 4, 3 * t.size // 4)

if not Path(f"output/table_statistics_hurst{hurst}_N{num_realizations}.csv").exists():
    _records = []
    for n in trange(num_realizations):
        # Square wave
        s = FBM(n=len(t), hurst=hurst, length=1).fbm()[: len(t)]
        s /= s.std()

        coeffs_ref = denoisers[0][1]._dwt(s)

        for a_type in ["square", "sawtooth"]:
            a = getattr(ss, a_type)((100 * t + np.random.randint(0, 99)) / np.pi)
            a /= a.std()

            x = s + scale * a

            for name, denoiser in denoisers:
                x_d, coeffs_d = denoiser.denoise(x, s, with_coeffs=True)

                # Wasserstein distance
                wasserstein_dist = [
                    stats.wasserstein_distance(cs_ref, cs_d)
                    for cs_ref, cs_d in zip(coeffs_ref, coeffs_d)
                ]

                # Fit Hurst exponent
                fs, Pxx = ss.welch(x_d, nperseg=1024)
                res = np.polyfit(np.log(fs)[1:151], np.log(Pxx)[1:151], 1)

                _records.append(
                    {
                        "artifact": a_type,
                        "denoiser": name,
                        "realization": n,
                        "scale": scale,
                        "mse": mse(x_d[mse_slice], s[mse_slice]),
                        "wasserstein": wasserstein_dist,
                        "num_w_scales": len(coeffs_d) - 1,
                        "hurst": -(res[0] + 1) / 2,
                    }
                )

    df = pd.DataFrame(_records)
    df.to_csv(f"output/table_statistics_hurst{hurst}_N{num_realizations}.csv")

df = pd.read_csv(f"output/table_statistics_hurst{hurst}_N{num_realizations}.csv")


df["avg_wasserstein"] = df.wasserstein.apply(lambda x: np.mean(ast.literal_eval(x)))
avg = df.groupby(["artifact", "denoiser", "scale"]).mean()
std = df.groupby(["artifact", "denoiser", "scale"]).std()

algorithms = {
    "wt_isoft": "Ideal ST",
    "wt_ihard": "Ideal HT",
    "wt_sure": "SureShrink",
    "wt_uni": "Universal WT",
    "wqn": "WQN",
}
metrics = {
    "mse": "MSE",
    "avg_wasserstein": "Avg Wasserstein",
    "hurst": "Hurst exponent",
}

table_tex = "\\toprule\n"
table_tex += " & " + " & ".join(algorithms.values()) + "\\\\\n"

for metric, metric_name in metrics.items():
    out = (
        metric_name
        + " (square) & "
        + " & ".join(
            f"{avg.loc[('square', algo, 2)][metric]:.2f}" for algo in algorithms
        )
    )
    out += "\\\\\n\\midrule\n"
    out += (
        metric_name
        + " (triangle) & "
        + " & ".join(
            f"{avg.loc[('sawtooth', algo, 2)][metric]:.2f}" for algo in algorithms
        )
    )
    out += "\\\\\n\\midrule\n"
    table_tex += out

table_tex += "\\bottomrule"

with open("./output/table_1.tex", "w") as f:
    f.write(table_tex)

print("Latex table written to `table_1.tex`.")
