FEATURE_CODE = "mom_signflip_12"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Smoothed Sign Flip Indicator (12)

    Description:
    Measures how frequently the return sign has flipped within the
    last 12 bars. High values indicate noisy or choppy price action;
    low values indicate steady trends.

    Formula:
    sign_t = sign(close_t - close_{t-1})
    flips_t = count(sign changes) over last 12 bars

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]
    close = g["close"]

    sign = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    flips = (sign != sign.shift(1)).astype(int)

    s = flips.rolling(12, min_periods=12).sum()
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
