FEATURE_CODE = "liq_flow_balance_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Flow Balance over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    buy_volume = g["volume"] * (g["close"] > g["open"]).astype(int)
    sell_volume = g["volume"] * (g["close"] < g["open"]).astype(int)

    lfb = (buy_volume - sell_volume).rolling(20, min_periods=20).sum()

    s = lfb.astype(float)
    s.name = FEATURE_CODE
    return s
