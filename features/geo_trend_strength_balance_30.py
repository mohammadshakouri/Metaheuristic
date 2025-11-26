FEATURE_CODE = "geo_trend_strength_balance_30"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Geometric Trend Strength Balance over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    up_moves = delta.clip(lower=0).rolling(30, min_periods=30).sum()
    down_moves = -delta.clip(upper=0).rolling(30, min_periods=30).sum()

    tsb = up_moves / down_moves.replace(0, np.nan)

    s = tsb.astype(float)
    s.name = FEATURE_CODE
    return s
