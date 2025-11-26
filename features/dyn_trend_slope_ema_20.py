FEATURE_CODE = "dyn_trend_slope_ema_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Slope of EMA(20) as trend speed indicator.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    ema20 = g["close"].ewm(span=20, adjust=False).mean()
    slope = ema20.diff()  # first difference as slope proxy

    s = slope.astype(float)
    s.name = FEATURE_CODE
    return s