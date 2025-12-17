FEATURE_CODE = "trend_dpo_30"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Detrended Price Oscillator (30)

    Description:
    Removes long-term trend by subtracting a lagged SMA(30) from price.
    Highlights short-term cycles and inflection points.

    Formula:
    offset = int(30/2 + 1)
    DPO_t = close_t - SMA(close,30) shifted by offset

    Input:
    OHLCV DataFrame

    Output:
    pd.Series(float), named FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]
    close = g["close"]

    sma = close.rolling(30, min_periods=30).mean()
    offset = int(30 / 2 + 1)

    shifted = sma.shift(offset)
    s = close - shifted

    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
