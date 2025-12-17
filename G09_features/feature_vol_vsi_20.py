FEATURE_CODE = "vol_vsi_20"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volume Shock Index (20)

    Description:
    Detects sudden volume spikes by comparing today's volume to the
    rolling mean and standard deviation of the last 20 bars.

    Formula:
    vsi = (volume - mean(volume,20)) / std(volume,20)

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    v = g["volume"].astype(float)
    mean = v.rolling(20, min_periods=20).mean()
    std = v.rolling(20, min_periods=20).std(ddof=0)

    s = (v - mean) / std.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
