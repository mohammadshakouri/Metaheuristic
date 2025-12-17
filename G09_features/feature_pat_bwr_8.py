FEATURE_CODE = "pat_bwr_8"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Body-Wick Ratio (8)

    Description:
    Measures the ratio of the candle's real body to its total wick
    length. Detects strong candles (high BWR) vs. indecision candles.

    Formula:
    body = |close - open|
    wick = (high - low) - body
    BWR = SMA(body,8) / SMA(wick,8)

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    body = (g["close"] - g["open"]).abs()
    wick = (g["high"] - g["low"]) - body

    b = body.rolling(8, min_periods=8).mean()
    w = wick.rolling(8, min_periods=8).mean()

    s = b / w.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
