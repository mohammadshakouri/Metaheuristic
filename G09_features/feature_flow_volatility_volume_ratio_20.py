FEATURE_CODE = "flow_volatility_volume_ratio_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Ratio of volume volatility to mean

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    std = g["volume"].rolling(20, min_periods=20).std(ddof=0)
    mean = g["volume"].rolling(20, min_periods=20).mean()
    s = std / mean.replace(0, np.nan)
    s = s.astype(float); s.name = FEATURE_CODE
    return s
