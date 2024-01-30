from statistics import mean, stdev

import numpy as np
import numpy.typing as npt
from sklearn.utils.extmath import randomized_svd


def choose_k(
    A_norm: npt.ArrayLike,
    K: int = 100,
    thresh: int = 6,
    noise_start: int = 80,
    q: int = 2,
    **kwargs,
) -> tuple:
    if K > np.min(A_norm.shape):
        msg = "For an m by n matrix, K must be smaller than the min(m,n)."
        raise ValueError(msg)
    if noise_start > (K - 5):
        msg = "There need to be at least 5 singular values considered noise."
        raise ValueError(msg)

    noise_svals = list(range(noise_start - 1, K))
    seed = kwargs.get("seed", None)
    _, d, _ = randomized_svd(M=A_norm, n_components=K, n_iter=q, random_state=seed)

    diffs = d[:-1] - d[1:]

    mu = mean(diffs[np.subtract(noise_svals, 1)])
    sigma = stdev(diffs[np.subtract(noise_svals, 1)])
    num_of_sds = (diffs - mu) / sigma

    k = np.max(np.where(num_of_sds > thresh)) + 1

    return {"k": k, "num_of_sds": num_of_sds, "d": d}
