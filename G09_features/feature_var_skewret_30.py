FEATURE_CODE = "var_skewret_30"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Rolling skewness of returns (30)

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    r = g["close"].pct_change()
    mean = r.rolling(30, min_periods=30).mean()
    std = r.rolling(30, min_periods=30).std(ddof=0)
    s = ((r - mean)**3).rolling(30, min_periods=30).mean() / (std**3)
    s = s.astype(float); s.name = FEATURE_CODE
    return s
