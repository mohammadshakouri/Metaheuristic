FEATURE_CODE = "reg_ewm_slope_40"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Exponential-Weighted Linear Regression Slope (40)

    Description:
    Computes the slope of a linear regression of close vs. time,
    where points inside the 40-bar window are exponentially weighted.
    Detects trend strength while giving more importance to recent data.

    Formula:
    slope_t = Cov_t(t, close) / Var_t(t)
    using exponential weights with span=40.

    Input:
    df: DataFrame with OHLCV columns (case-insensitive)

    Output:
    pd.Series(float), same index/length as df, name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]
    close = g["close"].astype(float)

    t = pd.Series(np.arange(len(close)), index=close.index, dtype=float)

    # Exponential weighted stats
    t_mean = t.ewm(span=40, adjust=False).mean()
    c_mean = close.ewm(span=40, adjust=False).mean()

    cov = ((t * close).ewm(span=40, adjust=False).mean()) - (t_mean * c_mean)
    var = ((t * t).ewm(span=40, adjust=False).mean()) - (t_mean**2)

    slope = cov / var.replace(0, np.nan)
    slope = slope.astype(float)
    slope.name = FEATURE_CODE
    return slope
