FEATURE_CODE = "geo_range_ratio_25"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Geometric Range Ratio over 25 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    high_low_range = g["high"] - g["low"]
    avg_range = high_low_range.rolling(25, min_periods=25).mean()

    range_ratio = high_low_range / avg_range.replace(0, np.nan)

    s = range_ratio.astype(float)
    s.name = FEATURE_CODE
    return s
