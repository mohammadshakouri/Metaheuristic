FEATURE_CODE = "trend_swing_efficiency_30"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Trend Swing Efficiency over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    swing_efficiency = delta.rolling(30, min_periods=30).apply(
        lambda x: np.sum(x[x > 0]) / -np.sum(x[x < 0]) if np.sum(x[x < 0]) != 0 else np.nan
    )

    s = swing_efficiency.astype(float)
    s.name = FEATURE_CODE
    return s
