FEATURE_CODE = "liq_turnover_rate_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Turnover Rate over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    turnover_rate = g["volume"] / g["volume"].rolling(20, min_periods=20).mean()

    s = turnover_rate.astype(float)
    s.name = FEATURE_CODE
    return s
