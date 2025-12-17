FEATURE_CODE = "flow_vol_sma_ratio_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Volume to SMA(volume) ratio

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    mean_vol = g["volume"].rolling(20, min_periods=20).mean()
    s = g["volume"] / mean_vol.replace(0, np.nan)
    s = s.astype(float); s.name = FEATURE_CODE
    return s
