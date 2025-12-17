FEATURE_CODE = "pat_rb_prop_20"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Real Body Proportion (20)

    Description:
    Measures the proportion of each candle composed of the real body
    relative to the full high–low range. Uses a 20-bar smoothing.

    Formula:
    body = |close - open|
    range = high - low
    rb_prop = SMA(body,20) / SMA(range,20)

    Input:
    OHLCV DataFrame

    Output:
    pd.Series(float), same index, named FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    body = (g["close"] - g["open"]).abs()
    rng = g["high"] - g["low"]

    b = body.rolling(20, min_periods=20).mean()
    r = rng.rolling(20, min_periods=20).mean()

    s = b / r.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
