FEATURE_CODE = "mom_ret_zscore_20"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Rolling Return Z-Score (20)

    Description:
    Computes the Z-score of the 1-bar returns over a 20-bar rolling
    window. Shows how extreme the current return is relative to recent
    volatility. Useful for anomaly detection and short-term momentum.

    Formula:
    r_t = close_t / close_{t-1} - 1
    z_t = (r_t - mean(r_{t-19:t})) / std(r_{t-19:t})

    Input:
    df: OHLCV DataFrame (case-insensitive)

    Output:
    pd.Series(float), same index/length as df, name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    ret = g["close"].pct_change()
    mean = ret.rolling(20, min_periods=20).mean()
    std = ret.rolling(20, min_periods=20).std(ddof=0)

    s = (ret - mean) / std.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
