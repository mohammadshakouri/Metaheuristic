FEATURE_CODE = "pat_wick_sym_12"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Wick Symmetry Index (12)

    Description:
    Compares upper and lower wick sizes over a 12-bar average.
    Values near 0 = symmetrical candles; large positive/negative =
    directional wicks.

    Formula:
    upper = high - max(open,close)
    lower = min(open,close) - low
    wsi = SMA(upper - lower,12)

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    upper = g["high"] - g[["open", "close"]].max(axis=1)
    lower = g[["open", "close"]].min(axis=1) - g["low"]

    diff = upper - lower
    s = diff.rolling(12, min_periods=12).mean()

    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
