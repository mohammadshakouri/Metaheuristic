FEATURE_CODE = "press_price_volume_impact"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Price impact proxy |Δclose| * volume

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    s = g["close"].diff().abs() * g["volume"]
    s = s.astype(float); s.name = FEATURE_CODE
    return s
