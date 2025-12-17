FEATURE_CODE = "risk_atr_14"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Average True Range (14)

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    tr = pd.concat([
    g["high"] - g["low"],
    (g["high"] - g["close"].shift()).abs(),
    (g["low"] - g["close"].shift()).abs()], axis=1).max(axis=1)
    s = tr.rolling(14, min_periods=14).mean()
    s = s.astype(float); s.name = FEATURE_CODE
    return s
