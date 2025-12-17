FEATURE_CODE = "pat_engulf_strength_5"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Engulfing Strength Score (5)

    Description:
    Detects engulfing patterns by comparing body size to the
    previous body size, smoothed over 5 bars.

    Formula:
    body_t = |close - open|
    strength = SMA(body_t / body_{t-1},5)

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    body = (g["close"] - g["open"]).abs()
    ratio = body / body.shift(1).replace(0, np.nan)

    s = ratio.rolling(5, min_periods=5).mean()
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
