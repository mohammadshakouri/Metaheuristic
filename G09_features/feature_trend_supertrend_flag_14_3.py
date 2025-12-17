FEATURE_CODE = "trend_supertrend_flag_14_3"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Simplified SuperTrend directional flag

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    tr = (g["high"] - g["low"]).combine((g["high"] - g["close"].shift()).abs(), max).combine((g["low"] - g["close"].shift()).abs(), max)
    atr = tr.rolling(14, min_periods=14).mean()
    hl2 = (g["high"] + g["low"]) / 2
    upper = hl2 + 3 * atr
    lower = hl2 - 3 * atr
    direction = np.where(g["close"] > upper.shift(), 1, np.where(g["close"] < lower.shift(), -1, np.nan))
    s = pd.Series(direction, index=g.index).ffill().astype(float)
    s.name = FEATURE_CODE
    return s
