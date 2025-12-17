FEATURE_CODE = "hybrid_price_volume_momentum_15"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Hybrid Price-Volume-Momentum Indicator over 15 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_change = g["close"].pct_change(15)
    vol_change = g["volume"].pct_change(15)
    momentum = g["close"].diff(15)

    pvm = (price_change * vol_change * momentum).rolling(15, min_periods=15).mean()

    s = pvm.astype(float)
    s.name = FEATURE_CODE
    return s
