FEATURE_CODE = "vol_hl_compression_15"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    High-Low Range Compression (15)

    Description:
    Measures how the current high–low range compares to the
    15-bar median range. Indicates volatility contraction (s < 1)
    or expansion (s > 1). Useful for regime detection.

    Formula:
    range_t = high_t - low_t
    med_range_t = median(range_{t-14:t})
    s_t = range_t / med_range_t

    Input:
    df: OHLCV DataFrame

    Output:
    pd.Series(float), aligned with df.index, name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    rng = g["high"] - g["low"]
    med = rng.rolling(15, min_periods=15).median()

    s = rng / med.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
