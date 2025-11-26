FEATURE_CODE = "hybrid_price_absorption_strength_25"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Hybrid Price Absorption Strength over 25 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_change = g["close"].diff()
    volume = g["volume"]

    absorption_strength = (price_change.abs() * volume).rolling(25, min_periods=25).sum() / volume.rolling(25, min_periods=25).sum().replace(0, np.nan)

    s = absorption_strength.astype(float)
    s.name = FEATURE_CODE
    return s
