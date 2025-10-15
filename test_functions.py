from programming_assignment_v3 import mean, detect_anomalies, filter_by_date
import pandas as pd
import numpy as np

def test_mean():
    assert mean([1, 2, 3, 4]) == np.mean([1, 2, 3, 4])

def test_detect_anomalies():
    df = pd.DataFrame({"CPU": [10, 12, 50, 11, 10]})
    df, anomalies = detect_anomalies(df, threshold=1)
    assert not anomalies.empty

def test_filter_by_date():
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2024-01-01", periods=3),
        "Air": [20, 21, 22],
        "CPU": [50, 52, 55]
    })
    result = filter_by_date(df, "2024-01-01", "2024-01-02")
    assert result[-1] == 2  # count = 2 rows
