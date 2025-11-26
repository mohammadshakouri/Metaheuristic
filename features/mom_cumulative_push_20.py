FEATURE_CODE = "mom_cumulative_push_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Cumulative Push over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    cumulative_push = delta.rolling(20, min_periods=20).sum()

    s = cumulative_push.astype(float)
    s.name = FEATURE_CODE
    return s
