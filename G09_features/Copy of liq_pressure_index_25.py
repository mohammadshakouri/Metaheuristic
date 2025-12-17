FEATURE_CODE = "liq_pressure_index_25"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Pressure Index over 25 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    buy_volume = g["volume"] * (g["close"] > g["open"]).astype(int)
    sell_volume = g["volume"] * (g["close"] < g["open"]).astype(int)

    lpi = (buy_volume - sell_volume).rolling(25, min_periods=25).sum() / g["volume"].rolling(25, min_periods=25).sum().replace(0, np.nan)

    s = lpi.astype(float)
    s.name = FEATURE_CODE
    return s
