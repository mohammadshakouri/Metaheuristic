FEATURE_CODE = "pat_cluster_5"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Up/Down Movement Cluster Length (5)

    Description:
    Measures whether recent price changes cluster into streaks
    of consecutive up or down moves. Uses the last 5 bars.
    Positive values = up cluster length; negative = down cluster length.
    Zero = mixed pattern.

    Formula:
    sign_t = sign(close_t - close_{t-1})
    cluster_t = sum of last 5 signs if all same sign, else 0

    Input:
    df: OHLCV DataFrame

    Output:
    pd.Series(float), name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]
    close = g["close"]

    sign = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    def cluster(arr):
        if len(arr) < 5:
            return np.nan
        s = arr.astype(int)
        if np.all(s == 1):
            return 5.0
        if np.all(s == -1):
            return -5.0
        return 0.0

    s = sign.rolling(5, min_periods=5).apply(cluster, raw=True)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
