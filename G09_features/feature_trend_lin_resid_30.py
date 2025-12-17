FEATURE_CODE = "trend_lin_resid_30"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Linear Trend Residual (30)

    Description:
    Measures deviation of price from a 30-bar rolling linear regression
    fit. High absolute values indicate trend breaks or reversals.

    Formula:
    fit_t = slope * t + intercept
    residual_t = close_t - fit_t

    Input: OHLCV DataFrame

    Output: pd.Series(float), name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]
    close = g["close"].astype(float)

    t = pd.Series(np.arange(len(close)), index=close.index, dtype=float)
    w = 30

    t_mean = t.rolling(w, min_periods=w).mean()
    c_mean = close.rolling(w, min_periods=w).mean()

    cov = (t * close).rolling(w, min_periods=w).mean() - t_mean * c_mean
    var = (t * t).rolling(w, min_periods=w).mean() - t_mean * t_mean

    slope = cov / var.replace(0, np.nan)
    intercept = c_mean - slope * t_mean

    fit = slope * t + intercept
    resid = close - fit

    resid = resid.astype(float)
    resid.name = FEATURE_CODE
    return resid
