FEATURE_CODE = "vol_range_volatility_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Range-Based Volatility over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    high_low_range = g["high"] - g["low"]
    rv = high_low_range.rolling(20, min_periods=20).std()

    s = rv.astype(float)
    s.name = FEATURE_CODE
    return s
