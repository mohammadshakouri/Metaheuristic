FEATURE_CODE = "mom_ntrm_16"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Normalized True Range Momentum (16)

    Description:
    Measures momentum in true range (volatility) changes, normalized
    by the mean ATR-like value.

    Formula:
    tr_t = max(high-low, |high-prev_close|, |low-prev_close|)
    atr_like = mean(tr,16)
    ntrm = (tr - tr.shift(1)) / atr_like

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    high, low, close = g["high"], g["low"], g["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_like = tr.rolling(16, min_periods=16).mean()

    s = (tr - tr.shift(1)) / atr_like.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
