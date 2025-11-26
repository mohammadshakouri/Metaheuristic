FEATURE_CODE = "geo_price_position_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Geometric Price Position over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    min_price = g["low"].rolling(20, min_periods=20).min()
    max_price = g["high"].rolling(20, min_periods=20).max()

    price_pos = (g["close"] - min_price) / (max_price - min_price).replace(0, np.nan)

    s = price_pos.astype(float)
    s.name = FEATURE_CODE
    return s
