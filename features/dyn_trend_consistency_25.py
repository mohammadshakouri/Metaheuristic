FEATURE_CODE = "dyn_trend_consistency_25"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Trend Consistency over 25 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    pos_days = (delta > 0).rolling(25, min_periods=25).sum()
    neg_days = (delta < 0).rolling(25, min_periods=25).sum()

    tc = pos_days / (pos_days + neg_days).replace(0, np.nan)

    s = tc.astype(float)
    s.name = FEATURE_CODE
    return s
