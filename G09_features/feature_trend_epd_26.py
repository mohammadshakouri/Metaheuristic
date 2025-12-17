FEATURE_CODE = "trend_epd_26"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Exponential Price Divergence (26)

    Description:
    Measures divergence between price and its EMA(26), normalized by
    ATR-like scale. Helps detect trend overextensions.

    Formula:
    ema = EMA(close,26)
    atr_like = SMA(|Δclose|,26)
    epd = (close - ema) / atr_like

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    close = g["close"]
    ema = close.ewm(span=26, adjust=False).mean()

    atr_like = close.diff().abs().rolling(26, min_periods=26).mean()

    s = (close - ema) / atr_like.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
