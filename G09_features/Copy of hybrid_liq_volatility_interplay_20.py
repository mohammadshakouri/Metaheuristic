FEATURE_CODE = "hybrid_liq_volatility_interplay_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Hybrid Liquidity-Volatility Interplay over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    volatility = g["close"].rolling(20, min_periods=20).std()
    avg_volume = g["volume"].rolling(20, min_periods=20).mean()

    lvi = (volatility * avg_volume).rolling(20, min_periods=20).mean()

    s = lvi.astype(float)
    s.name = FEATURE_CODE
    return s
