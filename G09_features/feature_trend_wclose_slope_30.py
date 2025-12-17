FEATURE_CODE = "trend_wclose_slope_30"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Weighted Close Trendline Slope (30)

    Description:
    Uses the weighted close price to compute trend slope over a 30-bar
    regression window. Gives more weight to price extremes.

    Formula:
    wclose = (high + low + close*2) / 4
    slope = linear regression slope over 30 bars

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    wc = (g["high"] + g["low"] + 2 * g["close"]) / 4
    t = pd.Series(np.arange(len(wc)), index=wc.index, dtype=float)

    w = 30
    t_mean = t.rolling(w, min_periods=w).mean()
    w_mean = wc.rolling(w, min_periods=w).mean()

    cov = (t * wc).rolling(w, min_periods=w).mean() - t_mean * w_mean
    var = (t * t).rolling(w, min_periods=w).mean() - t_mean * t_mean

    slope = cov / var.replace(0, np.nan)
    slope = slope.astype(float)
    slope.name = FEATURE_CODE
    return slope
