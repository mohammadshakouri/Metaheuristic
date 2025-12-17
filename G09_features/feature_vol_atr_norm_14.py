FEATURE_CODE = "vol_atr_norm_14"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    ATR Normalized by Median ATR (14)

    Description:
    Measures how current ATR compares to its rolling median.
    Values > 1 indicate volatility expansion.

    Formula:
    atr = SMA(TR,14)
    atr_norm = atr / median(atr,14)

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    high, low, close = g["high"], g["low"], g["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(14, min_periods=14).mean()
    med = atr.rolling(14, min_periods=14).median()

    s = atr / med.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
