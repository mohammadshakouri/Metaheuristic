FEATURE_CODE = "vol_vp_corr_20"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volume-Price Correlation (20)

    Description:
    Measures correlation between close price changes and volume over
    a 20-bar window. Indicates volume-confirmed moves.

    Formula:
    corr_t = Corr(Δclose, volume) over last 20 bars

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    dc = g["close"].diff()
    v = g["volume"].astype(float)

    s = dc.rolling(20, min_periods=20).corr(v)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
