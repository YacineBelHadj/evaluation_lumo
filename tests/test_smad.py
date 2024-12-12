import numpy as np
import pytest

from evaluation_lumo.metrics import mad

TEST_CASES = [
    # Test case 1: Balanced case with numpy array
    (np.array([1, 2, 3, 4, 5, 6]), 1.5),
    # multiple of the same value and the same MAD
    (np.array(2 * [1, 2, 3, 4, 5, 6]), 1.5),
    # negative values
    (np.array([-1, -2, -3, -4, -5, -6]), 1.5),
    # addition
    (np.array([1, 2, 3, 4, 5, 6]) + 10, 1.5),
    # constant value
    (np.array([1, 1, 1, 1, 1, 1]), 0),
]


@pytest.mark.parametrize("time_series, expected_smad", TEST_CASES)
def test_smad(time_series, expected_smad):
    if len(time_series) == 0:
        with pytest.raises(ValueError):
            mad(time_series)
    else:
        result = mad(time_series)
        assert result == pytest.approx(expected_smad, rel=1e-6)
