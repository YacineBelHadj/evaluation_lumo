import pandas as pd
import pytest

from evaluation_lumo.config import mat_state
from evaluation_lumo.utils import label_events


@pytest.fixture
def setup_data():
    # Sample timestamps for testing
    timestamps = pd.date_range(start="2020-08-01", end="2021-08-06", freq="D")

    # Define events as dictionaries
    events = {
        "healthytrain": {
            "start": "2020-08-01",
            "end": "2020-10-27T10:00:00",
            "description": "all damage mechanisms removed",
        },
        "healthytest": {
            "start": "2020-10-27",
            "end": "2020-11-09",
            "description": "healthy state after damage",
        },
        "damage1": {
            "start": "2020-11-09",
            "end": "2020-11-24",
            "description": "all damage mechanisms removed",
            "severity": "high",
            "location": "DAM4",
            "closest_sensor": 6,
        },
    }
    return timestamps, events


def test_label_events(setup_data):
    timestamps, events = setup_data

    # Apply the label_events function
    result_labels = label_events(timestamps, events)

    # Check that the result is a pandas Series
    assert isinstance(result_labels, pd.Series)

    # Check that the length of the result matches the input timestamps
    assert len(result_labels) == len(timestamps)

    # Check specific dates to ensure correct labeling
    test_cases = {
        "2020-08-02": "healthytrain",
        "2020-10-28": "healthytest",
        "2020-11-10": "damage1",
        "2021-01-01": "no_event",  # Assuming 0 indicates no event
    }

    for date_str, expected_label in test_cases.items():
        date = pd.Timestamp(date_str)
        label = result_labels[timestamps == date].values[0]
        assert label == expected_label


def test_label_events_2():
    timestamps = pd.date_range(
        start="2020-08-01", end="2022-01-20", freq="10min"
    )
    events = label_events(timestamps, mat_state)
    # count the number of each event
    res = events.value_counts()
    assert res["healthy_train"] == 8783
    assert res["healthy_test"] == 1204
    assert res["no_event"] == 25754
    assert res["damage1"] == 1975
    assert res["damage2"] == 2132
    assert res["damage3"] == 4730
    assert res["damage4"] == 2132
    assert res["damage5"] == 2414
    assert res["damage6"] == 2420
    assert res["healthy1"] == 1835
    assert res["healthy2"] == 16385
    assert res["healthy3"] == 1976
    assert res["healthy4"] == 1244
    assert res["healthy5"] == 1541
    assert res["healthy6"] == 2804
