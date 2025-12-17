FEATURE_CODE = "mom_rdi_40"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Return Drift Indicator (40)

    Description:
    Measures the long-term drift by comparing the cumulative return
    of the last 40 bars to the magnitude of price variation.

    Formula:
    drift = (close_t - close_{t-40}) / ATR_like(40)

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    close = g["close"]
    atr_like = close.diff().abs().rolling(40, min_periods=40).mean()

    s = (close - close.shift(40)) / atr_like.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
