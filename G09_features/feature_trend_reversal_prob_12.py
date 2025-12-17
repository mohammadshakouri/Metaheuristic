FEATURE_CODE = "trend_reversal_prob_12"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Reversal Probability Score (12)

    Description:
    Estimates likelihood of a trend reversal using the curvature of price:
    compares slope differences across two adjacent windows.

    Formula:
    slope1 = close_t - close_{t-6}
    slope2 = close_{t-6} - close_{t-12}
    prob = (slope1 - slope2) / (|slope1| + |slope2|)

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    c = g["close"]

    slope1 = c - c.shift(6)
    slope2 = c.shift(6) - c.shift(12)

    denom = (slope1.abs() + slope2.abs()).replace(0, np.nan)
    s = (slope1 - slope2) / denom

    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
