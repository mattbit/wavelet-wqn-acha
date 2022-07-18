import numpy as np
import scipy.signal as ss


def filter_bandpass(signal, low, high, fs, order=2):
    Wn = 2 * np.array([low, high]) / fs
    sos = ss.butter(order, Wn, btype="bandpass", output="sos")

    return ss.sosfiltfilt(sos, signal, axis=0)


def mask_to_intervals(mask, index=None):
    """Convert a boolean mask to a sequence of intervals.
    Caveat: when no index is given, the returned values correspond to the
    Python pure integer indexing (starting element included, ending element
    excluded). When an index is passed, pandas label indexing convention
    with strict inclusion is used.
    For example `mask_to_intervals([0, 1, 1, 0])` will return `[(1, 3)]`,
    but `mask_to_intervals([0, 1, 1, 0], ["a", "b", "c", "d"])` will return
    the value `[("b", "c")]`.
    Parameters
    ----------
    mask : numpy.ndarray
        A boolean array.
    index : Sequence, optional
        Elements to use as indices for determining interval start and end. If
        no index is given, integer array indices are used.
    Returns
    -------
    intervals : Sequence[Tuple[Any, Any]]
        A sequence of (start_index, end_index) tuples. Mindful of the caveat
        described above concerning the indexing convention.
    """
    if not np.any(mask):
        return []

    edges = np.flatnonzero(np.diff(np.pad(mask, 1)))
    intervals = edges.reshape((len(edges) // 2, 2))

    if index is not None:
        return [(index[i], index[j - 1]) for i, j in intervals]

    return [(i, j) for i, j in intervals]


def intervals_to_mask(intervals, size=None):
    mask = np.zeros(size, dtype=bool)
    for i, j in intervals:
        mask[i:j] = True

    return mask
