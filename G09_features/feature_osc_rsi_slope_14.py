FEATURE_CODE = "osc_rsi_slope_14"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Slope of RSI(14) indicator
    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    delta = g["close"].diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    period = 14
    gain = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    s = rsi.diff()
    s = s.astype(float); s.name = FEATURE_CODE
    return s
