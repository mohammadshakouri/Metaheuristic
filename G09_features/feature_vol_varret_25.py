FEATURE_CODE = "vol_varret_25"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Rolling Variance of Returns (25)

    Description:
    Measures the variance of 1-bar returns over a 25-bar window.
    Captures local volatility more sharply than ATR or range-based methods.

    Formula:
    r_t = pct_change(close)
    var_t = variance(r_{t-24:t})

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    r = g["close"].pct_change()
    s = r.rolling(25, min_periods=25).var(ddof=0)

    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
