FEATURE_CODE = "hybrid_volume_trend_conflict_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Hybrid Volume-Trend Conflict Indicator over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    trend = delta.rolling(20, min_periods=20).mean()
    avg_volume = g["volume"].rolling(20, min_periods=20).mean()

    conflict = (trend * (avg_volume - avg_volume.mean())).rolling(20, min_periods=20).mean()

    s = conflict.astype(float)
    s.name = FEATURE_CODE
    return s
