FEATURE_CODE = "liq_absorption_ratio_30"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Absorption Ratio over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_change = g["close"].diff()
    volume = g["volume"]

    lar = (price_change.abs() * volume).rolling(30, min_periods=30).sum() / volume.rolling(30, min_periods=30).sum().replace(0, np.nan)

    s = lar.astype(float)
    s.name = FEATURE_CODE
    return s
