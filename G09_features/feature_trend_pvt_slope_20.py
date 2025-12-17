FEATURE_CODE = "trend_pvt_slope_20"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Price-Volume Trend Slope (20)

    Description:
    Computes the slope of the Price-Volume Trend (PVT) using a
    20-bar rolling linear regression. Captures volume-weighted trend direction.

    Formula:
    PVT_t = PVT_{t-1} + (Δclose/close_{t-1}) * volume_t

    Output:
    pd.Series(float), name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    close, vol = g["close"], g["volume"]
    ret = close.pct_change()
    pvt = (ret * vol).cumsum()

    t = pd.Series(np.arange(len(pvt)), index=pvt.index, dtype=float)
    w = 20

    t_mean = t.rolling(w, min_periods=w).mean()
    p_mean = pvt.rolling(w, min_periods=w).mean()

    cov = (t * pvt).rolling(w, min_periods=w).mean() - t_mean * p_mean
    var = (t * t).rolling(w, min_periods=w).mean() - t_mean * t_mean

    slope = cov / var.replace(0, np.nan)
    slope = slope.astype(float)
    slope.name = FEATURE_CODE
    return slope
