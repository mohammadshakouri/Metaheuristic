FEATURE_CODE = "pat_crc_18"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Candle Range Compression Score (18)

    Description:
    Measures how compressed (narrow) the candles have become relative
    to their 18-bar average. Useful for breakout forecasting.

    Formula:
    range = high - low
    crc = range / SMA(range,18)

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    rng = g["high"] - g["low"]
    avg = rng.rolling(18, min_periods=18).mean()

    s = rng / avg.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
