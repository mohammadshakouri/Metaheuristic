FEATURE_CODE = "info_range_complexity_30"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Information Range Complexity over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_range = g["high"] - g["low"]
    complexity = price_range.rolling(30, min_periods=30).apply(
        lambda x: -np.sum((x / x.sum()) * np.log((x / x.sum()) + 1e-10))
    )

    s = complexity.astype(float)
    s.name = FEATURE_CODE
    return s
