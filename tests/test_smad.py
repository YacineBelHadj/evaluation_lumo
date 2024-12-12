import numpy as np
import pytest

from evaluation_lumo.metrics import smad

# Define test cases for SMAD
TEST_CASES_SMAD = [
    # Scenario 1: Normal case
    ([1, 2, 3, 4, 5], 0.3333333333333333),
    # Scenario 2: Robust to outliers
    ([1, 2, 3, 100, 5], 0.3333333333333333),
    # Scenario 3: Constant series (zero variability)
    ([3, 3, 3, 3, 3], 0.0),
    # Scenario 4: Scale invariance
    ([10, 20, 30, 40, 50], 0.3333333333333333),
    # Scenario 5: Single value in series
]


@pytest.mark.parametrize("time_series, expected_smad", TEST_CASES_SMAD)
def test_smad(time_series, expected_smad):
    """
    Test the SMAD function with various inputs to validate its behavior.
    """
    smad_value = smad(time_series)
    assert np.isclose(smad_value, expected_smad, atol=1e-6)
