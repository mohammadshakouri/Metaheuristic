FEATURE_CODE = "mom_ema_accel_14"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Smoothed Price Acceleration (EMA 14)

    Description:
    Measures price acceleration by comparing the difference between
    two EMAs: a fast EMA and a slow EMA. Acceleration indicates whether
    momentum is strengthening or weakening.

    Formula:
    fast = EMA(close, 7)
    slow = EMA(close, 14)
    accel = fast - slow

    Input:
    OHLCV DataFrame

    Output:
    pd.Series(float) named FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]
    close = g["close"]

    fast = close.ewm(span=7, adjust=False).mean()
    slow = close.ewm(span=14, adjust=False).mean()

    s = fast - slow
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
