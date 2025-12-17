FEATURE_CODE = "shift_close_leadlag_5"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Lead-lag momentum ratio (close / close.shift(5))

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    s = g["close"] / g["close"].shift(5)
    s = s.astype(float); s.name = FEATURE_CODE
    return s
