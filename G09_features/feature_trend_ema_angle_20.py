FEATURE_CODE = "trend_ema_angle_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """EMA(20) slope angle in radians

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    ema = g["close"].ewm(span=20, adjust=False).mean()
    slope = ema.diff()
    angle = np.arctan(slope / ema.shift(1))
    s = angle.astype(float); s.name = FEATURE_CODE
    return s
