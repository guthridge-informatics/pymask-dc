import importlib.resources as ir

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

from pyalra.choose_k import choose_k


@pytest.fixture
def random_matrix() -> npt.ArrayLike:
    matrix_file = ir.files("tests").joinpath("data", "choose_k", "random_matrix.csv")
    return np.loadtxt(matrix_file, delimiter=",")

@pytest.fixture
def expected_k() -> int:
    return 3

@pytest.fixture
def expected_num_of_sds() -> npt.ArrayLike:
    sds_file = ir.files("tests").joinpath("data", "choose_k", "num_sds.csv")
    return np.loadtxt(sds_file, delimiter=",")

@pytest.fixture
def expected_d() -> npt.ArrayLike:
    d_file = ir.files("tests").joinpath("data", "choose_k", "d.csv")
    return np.loadtxt(d_file, delimiter=",")

def test_choose_k(random_matrix, expected_k, expected_num_of_sds, expected_d):
    test_result = choose_k(A_norm=random_matrix, K=10, noise_start=5, seed=1)

    assert test_result["k"] == expected_k
    assert_allclose(test_result["num_of_sds"], expected_num_of_sds, atol=0.01)
    assert_allclose(test_result["d"], expected_d, atol=0.001)
