FEATURE_CODE = "geo_body_wick_ratio"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Geometric Body to Wick Ratio
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    body = (g["close"] - g["open"]).abs()
    upper_wick = g["high"] - np.maximum(g["close"], g["open"])
    lower_wick = np.minimum(g["close"], g["open"]) - g["low"]
    total_wick = upper_wick + lower_wick

    gbwr = body / total_wick.replace(0, np.nan)

    s = gbwr.astype(float)
    s.name = FEATURE_CODE
    return s
