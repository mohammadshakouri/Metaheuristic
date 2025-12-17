FEATURE_CODE = "mom_williamsR_14"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Williams %R indicator

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    high_max = g["high"].rolling(14, min_periods=14).max()
    low_min = g["low"].rolling(14, min_periods=14).min()
    s = -100 * (high_max - g["close"]) / (high_max - low_min).replace(0, np.nan)
    s = s.astype(float); s.name = FEATURE_CODE
    return s
