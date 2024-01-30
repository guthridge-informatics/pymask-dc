import numpy as np
import numpy.typing as npt
import scipy as sp
from loguru import logger
from sklearn.utils.extmath import randomized_svd

from pyalra.choose_k import choose_k


def alra(
    a_norm: npt.ArrayLike,
    k: int = 0,
    q: int = 10,
    quantile_prob: float = 0.001,
    seed: float | None = None,
) -> dict[str, npt.ArrayLike]:
    logger.info(f"Read matrix with {a_norm.shape[0]} cells and {a_norm.shape[1]} genes")

    if k == 0:
        k, *_ = choose_k(a_norm)
        logger.info(f"Chose k = {k}")

    logger.info("Getting nonzeros\n")
    originally_nonzero = a_norm > 0

    u, d, v = randomized_svd(M=a_norm, n_components=k, n_iter=q, random_state=seed)

    a_norm_rank_k = np.matmul(np.matmul(u[:, :k], np.diag(d[:k])), v[:k, :])

    logger.info(f"Find the {quantile_prob} quantile of each gene")
    a_norm_rank_k_mins = abs(np.apply_along_axis(np.quantile, 0, a_norm_rank_k, q=0.001))

    logger.info("Sweep")
    a_norm_rank_k_cor = a_norm_rank_k.copy()
    a_norm_rank_k_cor[a_norm_rank_k <= np.tile(a_norm_rank_k_mins, (len(a_norm_rank_k), 1))] = 0

    sigma_1 = np.apply_along_axis(sp.stats.tstd, 0, a_norm_rank_k_cor, nan_policy="omit")
    sigma_2 = np.apply_along_axis(sp.stats.tstd, 0, a_norm, nan_policy="omit")
    mu_1 = np.sum(a_norm_rank_k_cor, axis=0) / np.sum(a_norm_rank_k_cor > 0, axis=0)
    mu_2 = np.sum(a_norm, axis=0) / np.sum(a_norm > 0, axis=0)

    toscale = np.logical_and(
        np.logical_and(~np.isnan(sigma_1), ~np.isnan(sigma_2)),
        np.logical_and(~np.logical_and((sigma_1 == 0), (sigma_2 == 0)), ~(sigma_1 == 0)),
    )

    logger.info(f"Scaling all except for {sum(~toscale)} columns")

    sigma_1_2 = sigma_2 / sigma_1
    toadd = -1 * mu_1 * sigma_2 / sigma_1 + mu_2

    a_norm_rank_k_temp = a_norm_rank_k_cor[:, toscale].copy()
    a_norm_rank_k_temp = np.multiply(a_norm_rank_k_temp, sigma_1_2[toscale])
    a_norm_rank_k_temp = np.add(a_norm_rank_k_temp, toadd[toscale])

    a_norm_rank_k_cor_sc = a_norm_rank_k_cor.copy()
    a_norm_rank_k_cor_sc[:, toscale] = a_norm_rank_k_temp
    a_norm_rank_k_cor_sc[a_norm_rank_k_cor == 0] = 0

    lt0 = a_norm_rank_k_cor_sc < 0
    a_norm_rank_k_cor_sc[lt0] = 0

    a_norm_size = a_norm.shape[0] * a_norm.shape[1]
    logger.info(
        f"{100*sum(lt0)/a_norm_size:.2f}% of the values became negative in the scaling process and were set to zero"
    )

    nonzero_mask = np.logical_and(originally_nonzero, (a_norm_rank_k_cor_sc == 0))
    a_norm_rank_k_cor_sc[nonzero_mask] = a_norm[nonzero_mask]

    original_nz = np.sum(a_norm > 0) / a_norm_size
    completed_nz = np.sum(a_norm_rank_k_cor_sc > 0) / a_norm_size
    logger.info(f"The matrix went from {100*original_nz:.2f}% nonzero to {100*completed_nz:.2f}% nonzero")

    return {
        "A_norm_rank_k": a_norm_rank_k,
        "A_norm_rank_k_cor": a_norm_rank_k_cor,
        "A_norm_rank_k_cor_sc": a_norm_rank_k_cor_sc,
    }
