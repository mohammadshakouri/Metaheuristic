FEATURE_CODE = "vol_mmvd_25"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Mean-Median Volatility Divergence (25)

    Description:
    Compares the rolling mean and rolling median of absolute returns
    to detect skewed or abnormal volatility distributions.

    Formula:
    absret = |close_t - close_{t-1}|
    s_t = mean(absret,25) - median(absret,25)

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    absret = g["close"].diff().abs()

    mean = absret.rolling(25, min_periods=25).mean()
    med = absret.rolling(25, min_periods=25).median()

    s = mean - med
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
