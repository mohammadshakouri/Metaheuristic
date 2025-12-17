FEATURE_CODE = "mom_ret_accel_12"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Smoothed Return Acceleration (12)

    Description:
    Computes acceleration of returns by taking the difference between
    a fast EMA of returns and a slow EMA. Highlights momentum turning points.

    Formula:
    r_t = pct_change(close)
    accel_t = EMA(r,6) - EMA(r,12)

    Output:
    pd.Series(float), same index and length, name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    r = g["close"].pct_change()
    fast = r.ewm(span=6, adjust=False).mean()
    slow = r.ewm(span=12, adjust=False).mean()

    s = fast - slow
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
