FEATURE_CODE = "liq_volatility_corr_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity-Volatility Correlation over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    volatility = g["close"].rolling(20, min_periods=20).std()
    avg_volume = g["volume"].rolling(20, min_periods=20).mean()

    lvc = volatility.rolling(20, min_periods=20).corr(avg_volume)

    s = lvc.astype(float)
    s.name = FEATURE_CODE
    return s
