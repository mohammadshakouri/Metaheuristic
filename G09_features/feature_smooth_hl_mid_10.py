FEATURE_CODE = "smooth_hl_mid_10"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Rolling mean of mid price ((high+low)/2)

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    mid = (g["high"] + g["low"]) / 2
    s = mid.rolling(10, min_periods=10).mean()
    s = s.astype(float); s.name = FEATURE_CODE
    return s
