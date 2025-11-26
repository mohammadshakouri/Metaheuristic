FEATURE_CODE = "vol_vol_of_vol_50"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volatility of Volatility over 50 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    volatility = g["close"].rolling(20, min_periods=20).std()
    vov = volatility.rolling(50, min_periods=50).std()

    s = vov.astype(float)
    s.name = FEATURE_CODE
    return s
