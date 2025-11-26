FEATURE_CODE = "mom_vol_weighted_roc_15"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volume-Weighted Rate of Change over 15 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_change = g["close"].pct_change(15)
    vol_weight = g["volume"] / g["volume"].rolling(15, min_periods=15).mean()

    vwr = price_change * vol_weight

    s = vwr.astype(float)
    s.name = FEATURE_CODE
    return s
