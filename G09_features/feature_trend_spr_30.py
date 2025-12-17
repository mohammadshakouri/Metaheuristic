FEATURE_CODE = "trend_spr_30"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Smoothed Price Ratio (30)

    Description:
    Ratio between price and its 30-bar SMA. Values > 1 indicate
    price above trend; < 1 indicates below trend.

    Formula:
    spr_t = close_t / SMA(close,30)

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    close = g["close"]
    sma = close.rolling(30, min_periods=30).mean()

    s = close / sma.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
