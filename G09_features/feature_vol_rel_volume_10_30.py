FEATURE_CODE = "vol_rel_volume_10_30"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Relative Volume Spike Score (10, 30)

    Description:
    Measures volume acceleration by comparing a 10-bar average volume
    to a slower 30-bar average volume. Values > 1 indicate an abnormal
    volume spike relative to recent history.

    Formula:
    rv_t = SMA(volume,10) / SMA(volume,30)

    Input:
    df: DataFrame with OHLCV columns (case-insensitive)

    Output:
    pd.Series(float), same index/length as df, name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]
    v = g["volume"].astype(float)

    fast = v.rolling(10, min_periods=10).mean()
    slow = v.rolling(30, min_periods=30).mean()

    s = fast / slow.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
