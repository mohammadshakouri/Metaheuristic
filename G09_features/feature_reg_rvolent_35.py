FEATURE_CODE = "reg_rvolent_35"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Return Volatility Entropy (35)

    Description:
    Computes Shannon entropy of the absolute-return direction
    (vol rising, flat, falling) over 35 bars. Measures volatility regime stability.

    Formula:
    v = |Δclose|
    sign = sign(v - v.shift(1))
    entropy = -sum(p * log2(p))

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    v = g["close"].diff().abs()
    sign = (v - v.shift(1)).apply(lambda x: -1 if x < 0 else (1 if x > 0 else 0))

    def ent(arr):
        n = len(arr)
        if n == 0:
            return np.nan
        counts = [np.mean(arr == k) for k in [-1, 0, 1]]
        p = np.array([x for x in counts if x > 0])
        return -np.sum(p * (np.log(p) / np.log(2)))

    s = sign.rolling(35, min_periods=35).apply(ent, raw=True)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
