FEATURE_CODE = "reg_skewret_50"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Rolling Skewness of Returns (50)

    Description:
    Computes skewness of log returns over a 50-bar window.
    Helps detect asymmetric volatility regimes.

    Formula:
    logret = log(close_t / close_{t-1})
    skewness_t = rolling skewness(logret,50)

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    logret = np.log(g["close"] / g["close"].shift(1))

    def skew(arr):
        m = arr.mean()
        sd = arr.std()
        if sd == 0:
            return np.nan
        return np.mean(((arr - m) / sd) ** 3)

    s = logret.rolling(50, min_periods=50).apply(skew, raw=True)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
