FEATURE_CODE = "ent_vol_sign_50"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volatility Direction Entropy (50)

    Description:
    Computes the Shannon entropy of volatility-direction signs over
    the last 50 bars. Measures how predictable volatility rises/falls
    have been. Low entropy = stable regime; high entropy = chaotic.

    Formula:
    vol_t = |close_t - close_{t-1}|
    sign_t = sign(vol_t - vol_{t-1})
    entropy = -sum(p * log2(p)), p = frequency of {-1,0,1}

    Input:
    OHLCV DataFrame

    Output:
    pd.Series(float), name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    v = g["close"].diff().abs()
    sign = (v - v.shift(1)).apply(lambda x: -1 if x < 0 else (1 if x > 0 else 0))

    def ent_window(arr):
        n = len(arr)
        if n == 0:
            return np.nan
        p_neg = np.count_nonzero(arr == -1) / n
        p_zero = np.count_nonzero(arr == 0) / n
        p_pos = np.count_nonzero(arr == 1) / n
        p = np.array([p_neg, p_zero, p_pos], dtype=float)
        p = p[p > 0]
        return -np.sum(p * (np.log(p) / np.log(2)))

    s = sign.rolling(50, min_periods=50).apply(ent_window, raw=True)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
