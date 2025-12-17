FEATURE_CODE = "vol_iqr_20"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Rolling Interquartile Range of Close (20)

    Description:
    Measures dispersion of closing prices using the IQR instead of
    standard deviation. Robust to outliers.

    Formula:
    IQR_t = Q3(close_{t-19:t}) - Q1(close_{t-19:t})

    Input:
    OHLCV DataFrame

    Output:
    pd.Series(float), name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]
    close = g["close"]

    def iqr(arr):
        return np.percentile(arr, 75) - np.percentile(arr, 25)

    s = close.rolling(20, min_periods=20).apply(iqr, raw=True)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
