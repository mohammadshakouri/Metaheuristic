FEATURE_CODE = "trend_mm_mid_50"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Min-Max Midline Distance (50)

    Description:
    Measures how far price is above or below the midpoint of the
    50-bar rolling high-low channel. Useful for identifying trend
    strength without using moving averages.

    Formula:
    mid = (rolling_max + rolling_min) / 2
    dist = close - mid

    Input:
    OHLCV DataFrame

    Output:
    pd.Series(float), name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    high, low, close = g["high"], g["low"], g["close"]

    roll_max = high.rolling(50, min_periods=50).max()
    roll_min = low.rolling(50, min_periods=50).min()
    mid = (roll_max + roll_min) / 2

    s = close - mid
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
