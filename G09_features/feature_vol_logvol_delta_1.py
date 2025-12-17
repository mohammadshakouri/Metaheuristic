FEATURE_CODE = "vol_logvol_delta_1"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Log-Volatility Change (1-bar)

    Description:
    Computes the 1-bar change in log volatility (absolute returns).
    Highlights sudden jumps in volatility.

    Formula:
    vol_t = |close_t - close_{t-1}|
    log_vol_t = log(vol_t)
    s_t = log_vol_t - log_vol_{t-1}

    Input:
    OHLCV DataFrame

    Output:
    pd.Series(float), name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    v = g["close"].diff().abs()
    logv = np.log(v.replace(0, np.nan))

    s = logv.diff()
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
