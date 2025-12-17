FEATURE_CODE = "speed_close_zscore_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Z-score of close prices over 20 bars

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    mean = g["close"].rolling(20, min_periods=20).mean()
    std = g["close"].rolling(20, min_periods=20).std(ddof=0)
    s = (g["close"] - mean) / std.replace(0, np.nan)
    s = s.astype(float); s.name = FEATURE_CODE
    return s
