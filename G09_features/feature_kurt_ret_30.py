FEATURE_CODE = "kurt_ret_30"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Rolling kurtosis of returns (30)

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    r = g["close"].pct_change()
    mean = r.rolling(30, min_periods=30).mean()
    std = r.rolling(30, min_periods=30).std(ddof=0)
    s = ((r - mean)**4).rolling(30, min_periods=30).mean() / (std**4)
    s = s.astype(float); s.name = FEATURE_CODE
    return s
