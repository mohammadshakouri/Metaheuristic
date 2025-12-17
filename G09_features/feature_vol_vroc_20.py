FEATURE_CODE = "vol_vroc_20"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volume Rate of Change (20)

    Description:
    Measures percentage change in volume relative to 20 bars ago.
    Detects surges or drops in trading activity.

    Formula:
    VROC_t = (volume_t - volume_{t-20}) / volume_{t-20}

    Input:
    OHLCV DataFrame

    Output:
    pd.Series(float), name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    v = g["volume"].astype(float)

    s = (v - v.shift(20)) / v.shift(20).replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
