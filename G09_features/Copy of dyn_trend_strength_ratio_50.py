FEATURE_CODE = "dyn_trend_strength_ratio_50"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Trend Strength Ratio over 50 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    up = delta.clip(lower=0).rolling(50, min_periods=50).sum()
    down = -delta.clip(upper=0).rolling(50, min_periods=50).sum()

    tsr = up / down.replace(0, np.nan)

    s = tsr.astype(float)
    s.name = FEATURE_CODE
    return s
