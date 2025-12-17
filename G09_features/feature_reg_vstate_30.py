FEATURE_CODE = "reg_vstate_30"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volatility State Score (30)

    Description:
    Computes a normalized indicator of whether absolute returns
    are in a low, medium, or high volatility regime, based on where
    they fall between rolling min and max.

    Formula:
    v = |Δclose|
    score = (v - min(v,30)) / (max(v,30) - min(v,30))

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    v = g["close"].diff().abs()

    v_min = v.rolling(30, min_periods=30).min()
    v_max = v.rolling(30, min_periods=30).max()

    s = (v - v_min) / (v_max - v_min).replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
