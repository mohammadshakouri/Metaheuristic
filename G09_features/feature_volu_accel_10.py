FEATURE_CODE = "volu_accel_10"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Second difference (acceleration) of volume

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    s = g["volume"].diff().diff()
    s = s.astype(float); s.name = FEATURE_CODE
    return s
    