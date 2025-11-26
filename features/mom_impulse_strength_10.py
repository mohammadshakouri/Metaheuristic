FEATURE_CODE = "mom_impulse_strength_10"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Impulse Strength over 10 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    momentum = g["close"].diff(10)
    volatility = g["close"].rolling(10, min_periods=10).std()

    mis = momentum / volatility.replace(0, np.nan)

    s = mis.astype(float)
    s.name = FEATURE_CODE
    return s
