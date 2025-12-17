FEATURE_CODE = "vol_vcr_15"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volume Compression Ratio (15)

    Description:
    Measures how compressed (low) or expanded (high) volume is relative
    to its 15-bar rolling max. Values near 0 = compressed; near 1 = expanded.

    Formula:
    VCR_t = volume_t / max(volume_{t-14:t})

    Input:
    OHLCV DataFrame

    Output:
    pd.Series(float), named FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    v = g["volume"].astype(float)
    v_max = v.rolling(15, min_periods=15).max()

    s = v / v_max.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
